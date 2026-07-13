#!/usr/bin/env python3
"""Autonomous navigate-and-pick.

Builds on the mobile search-and-pick, but replaces the naive "spin then drive
straight at the box" long-range motion with real Nav2 goal-based navigation
(global path planning + costmap obstacle avoidance). Once Nav2 has driven the
base to a pose ~APPROACH_DIST in front of the box, control is handed off to
the already-verified visual servo (SearchAndPick.search_and_approach) for the
final precise positioning and then the inherited arm run() pick sequence.

Flow:
  1) Arm -> search pose; spin in place (high-priority /cmd_vel_search via
     twist_mux) until the box is first detected.
  2) Transform the box from base_link into the map frame (slam_toolbox
     supplies map->odom; diff_drive supplies odom->base_link).
  3) Compute an approach goal ~APPROACH_DIST from the box, on the robot->box
     ray, facing the box; send it as a Nav2 NavigateToPose goal. Nav2 plans a
     path and follows it, steering around obstacles (base-level avoidance).
  4) On arrival, run the inherited visual search_and_approach() to servo the
     last short distance precisely, then run() to scan/grasp/carry/place.

The arm keeps its existing MoveIt collision-aware planning unchanged.
"""
import math
import time
import threading

import rclpy
from rclpy.action import ActionClient
from geometry_msgs.msg import Twist, PoseStamped, PointStamped
import tf2_ros
import tf2_geometry_msgs  # registers do_transform_point for PointStamped

from nav2_msgs.action import NavigateToPose

from pickplace_arm_bringup.pick_and_place import scan_quat
from pickplace_arm_bringup.search_and_pick import (
    SearchAndPick, SEARCH_POSITION, SEARCH_PITCH, SPIN_ANGULAR, SPIN_STEP_RAD,
    SPIN_SETTLE_SEC, SPIN_STEPS_PER_REV)

# How far in front of the box Nav2 should stop. Left deliberately loose --
# Nav2 arrives within its goal tolerance (~0.2 m), then the visual servo
# closes the remaining distance to the exact grasp stop distance. This value
# keeps the box comfortably inside the search pose's [0.45,1.1] m detection
# band on arrival.
APPROACH_DIST = 0.75
NAV_TIMEOUT_SEC = 120.0


class NavAndPick(SearchAndPick):
    def __init__(self):
        super().__init__()
        # Route wheel commands through twist_mux (high-priority input) instead
        # of straight to the controller, so Nav2 and this node share the base.
        self.destroy_publisher(self.cmd_vel_pub)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel_search', 10)
        self.nav_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')
        self.get_logger().info('Nav-and-pick node ready')

    # --- initial detection: spin in place until the box is first seen -------
    def spin_until_seen(self, timeout_sec=90.0):
        log = self.get_logger()
        sx, sy, sz = SEARCH_POSITION
        self.move_pose(sx, sy, sz, label='search-scan',
                       quat_xyzw=scan_quat(SEARCH_PITCH))
        deadline = time.time() + timeout_sec
        steps = 0
        while time.time() < deadline:
            det = self.detect_box_pose(timeout_sec=1.0)
            if det is not None:
                bx, by, _ = det
                log.info(f'[nav] box first seen: dist={math.hypot(bx,by):.2f}m '
                          f'bearing={math.degrees(math.atan2(by,bx)):.1f}deg')
                return det
            self._rotate_step(SPIN_STEP_RAD)
            steps += 1
            if steps >= SPIN_STEPS_PER_REV:
                log.info('[nav] full rotation, nothing found -- creeping fwd')
                self._drive_blind(0.2, 2.0)
                time.sleep(SPIN_SETTLE_SEC)
                steps = 0
        return None

    # --- transform a base_link point into the map frame ---------------------
    def box_in_map(self, bx, by):
        try:
            tf = self.tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=2.0))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            self.get_logger().error(f'[nav] map<-base_link TF failed: {e}')
            return None
        pt = PointStamped()
        pt.header.frame_id = 'base_link'
        pt.point.x, pt.point.y, pt.point.z = bx, by, 0.0
        pm = tf2_geometry_msgs.do_transform_point(pt, tf)
        return pm.point.x, pm.point.y

    def robot_in_map(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=2.0))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            self.get_logger().error(f'[nav] robot pose TF failed: {e}')
            return None
        t = tf.transform.translation
        return t.x, t.y

    def compute_approach_goal(self, box_map, robot_map):
        bx, by = box_map
        rx, ry = robot_map
        dx, dy = bx - rx, by - ry
        d = math.hypot(dx, dy)
        if d < 1e-3:
            ux, uy = 1.0, 0.0
        else:
            ux, uy = dx / d, dy / d
        gx = bx - APPROACH_DIST * ux
        gy = by - APPROACH_DIST * uy
        yaw = math.atan2(uy, ux)  # face the box
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.pose.position.x = gx
        goal.pose.position.y = gy
        goal.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.orientation.w = math.cos(yaw / 2.0)
        return goal

    def navigate_to(self, goal_pose, timeout_sec=NAV_TIMEOUT_SEC):
        log = self.get_logger()
        if not self.nav_client.wait_for_server(timeout_sec=10.0):
            log.error('[nav] NavigateToPose action server unavailable')
            return False
        goal_msg = NavigateToPose.Goal()
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose = goal_pose
        log.info(f'[nav] sending Nav2 goal '
                  f'({goal_pose.pose.position.x:.2f},'
                  f'{goal_pose.pose.position.y:.2f})')
        send_fut = self.nav_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_fut, timeout_sec=10.0)
        handle = send_fut.result()
        if handle is None or not handle.accepted:
            log.error('[nav] Nav2 goal rejected')
            return False
        result_fut = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_fut, timeout_sec=timeout_sec)
        if result_fut.result() is None:
            log.error('[nav] Nav2 goal timed out')
            return False
        log.info('[nav] Nav2 reported arrival')
        return True

    def run_autonomous(self):
        log = self.get_logger()
        log.info('=== NAV AND PICK: START ===')

        det = self.spin_until_seen()
        if det is None:
            log.error('Box never detected -- aborting.')
            return
        bx, by, _ = det

        box_map = self.box_in_map(bx, by)
        robot_map = self.robot_in_map()
        if box_map is None or robot_map is None:
            log.error('TF unavailable -- cannot build Nav2 goal.')
            return
        log.info(f'[nav] box in map: ({box_map[0]:.2f},{box_map[1]:.2f})')

        goal = self.compute_approach_goal(box_map, robot_map)
        if not self.navigate_to(goal):
            log.error('Navigation failed -- aborting pick.')
            return

        # Nav2 got us close; visually servo the final short distance precisely,
        # then run the verified scan/grasp/carry/place sequence.
        if self.search_and_approach():
            self.run()
        else:
            log.error('Visual approach failed after navigation -- no pick.')


def main():
    rclpy.init()
    node = NavAndPick()
    ex = rclpy.executors.MultiThreadedExecutor(4)
    ex.add_node(node)

    def task():
        time.sleep(3.0)
        node.run_autonomous()

    t = threading.Thread(target=task, daemon=True)
    t.start()
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
