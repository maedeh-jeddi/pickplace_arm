#!/usr/bin/env python3
"""Full autonomous warehouse mission.

State machine (localizing on the saved map with AMCL, navigating with Nav2):
  INIT     wait for AMCL localization (map -> base_link TF).
  SEARCH   patrol the map via Nav2 NavigateThroughPoses (drive between
           locations WITHOUT stopping) while the base-mounted FRONT camera
           watches for the box; on a stable detection, cancel the patrol.
  APPROACH transform the box into the map frame, Nav2 to a pose ~0.6 m in
           front of it, then the wrist-camera visual servo for the final
           precise positioning.
  PICK     arm picks the box up and holds it in the carry pose.
  DELIVER  Nav2 to the pre-defined delivery point, carrying the box.
  PLACE    arm puts the box down and returns home.
  PARK     Nav2 to the parking station and stop.

Builds on NavAndPick (navigate_to / box_in_map / make_map_goal /
search_and_approach) and PickAndPlace (detect_box_front / pick_up_box /
place_box_down).
"""
import math
import time
import threading

import rclpy
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped
import tf2_ros

from nav2_msgs.action import NavigateThroughPoses

from pickplace_arm_bringup.nav_and_pick import NavAndPick, APPROACH_DIST

# --- mission targets (map frame; map origin = robot's mapping start pose) -----
# Patrol route the robot sweeps while watching for the box (open lanes in the
# warehouse; each yaw faces the next leg so the front camera looks ahead).
# NOTE: NavigateThroughPoses treats the LAST pose as the goal, so the route must
# NOT end at the robot's current spot (else it "arrives" instantly). It ends
# far from the (0,0) start, and re-patrols reverse direction (see
# search_via_patrol) so a repeat lap also has a distant final goal.
PATROL_WAYPOINTS = [
    (2.5, 0.0), (2.5, -3.5), (-1.0, -4.0), (-3.5, -1.5),
    (-3.5, 1.5), (0.0, 2.0), (2.0, 2.0),
]
DELIVERY_POSE = (-4.0, 2.0, 0.0)     # x, y, yaw -- where the box is dropped off
PARKING_POSE = (4.0, -4.0, 0.0)      # x, y, yaw -- final parking station

FRONT_DETECT_MAX_DIST = 2.5   # only act on front-cam detections within this range
FRONT_DETECT_CONSEC = 2       # consecutive detections before committing
SEARCH_TIMEOUT_SEC = 240.0


class Mission(NavAndPick):
    def __init__(self):
        super().__init__()
        self.tp_client = ActionClient(
            self, NavigateThroughPoses, '/navigate_through_poses')
        self.get_logger().info('Mission node ready')

    # --- helpers ------------------------------------------------------------
    def _patrol_poses(self, reverse=False):
        wps = list(reversed(PATROL_WAYPOINTS)) if reverse else list(PATROL_WAYPOINTS)
        poses = []
        prev_yaw = 0.0
        for i, (x, y) in enumerate(wps):
            # face the next leg; the final pose keeps the previous heading
            if i + 1 < len(wps):
                nx, ny = wps[i + 1]
                prev_yaw = math.atan2(ny - y, nx - x)
            poses.append(self.make_map_goal(x, y, prev_yaw))
        return poses

    def wait_for_localization(self, timeout_sec=60.0):
        log = self.get_logger()
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            try:
                self.tf_buffer.lookup_transform(
                    'map', 'base_link', rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=1.0))
                log.info('[mission] localized (map->base_link available)')
                return True
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException):
                time.sleep(0.5)
        log.error('[mission] no map->base_link TF -- is AMCL running?')
        return False

    def _cancel(self, handle):
        try:
            fut = handle.cancel_goal_async()
            rclpy.spin_until_future_complete(self, fut, timeout_sec=5.0)
        except Exception:
            pass
        self._stop_base()

    # --- SEARCH: patrol (no stopping) while the front camera watches ---------
    def search_via_patrol(self, timeout_sec=SEARCH_TIMEOUT_SEC):
        log = self.get_logger()
        log.info('=== MISSION SEARCH: patrol + front-camera watch ===')
        # tuck the arm compactly (home) so it stays within the footprint and
        # doesn't block the forward view while driving.
        self.arm.move_to_configuration([0.0] * 6)
        self.arm.wait_until_executed()

        if not self.tp_client.wait_for_server(timeout_sec=10.0):
            log.error('[mission] NavigateThroughPoses server unavailable')
            return None

        def send_patrol(reverse):
            g = NavigateThroughPoses.Goal()
            g.poses = self._patrol_poses(reverse)
            fut = self.tp_client.send_goal_async(g)
            rclpy.spin_until_future_complete(self, fut, timeout_sec=10.0)
            h = fut.result()
            if h is None or not h.accepted:
                return None, None
            return h, h.get_result_async()

        reverse = False
        handle, result_fut = send_patrol(reverse)
        if handle is None:
            log.error('[mission] patrol goal rejected')
            return None

        consec = 0
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            det = self.detect_box_front(timeout_sec=1.0)
            if det is not None and math.hypot(det[0], det[1]) < FRONT_DETECT_MAX_DIST:
                consec += 1
                log.info(f'[mission] front camera sees box: '
                          f'dist={math.hypot(det[0], det[1]):.2f}m ({consec})')
                if consec >= FRONT_DETECT_CONSEC:
                    log.info('=== MISSION SEARCH: box found, stopping patrol ===')
                    self._cancel(handle)
                    return det
            else:
                consec = 0
            if result_fut.done():   # lap finished without finding -> reverse + re-patrol
                log.info('[mission] patrol lap done, no box -- reversing route')
                reverse = not reverse
                handle, result_fut = send_patrol(reverse)
                if handle is None:
                    return None
                time.sleep(1.0)     # let the new lap start driving before re-checking
        self._cancel(handle)
        log.error('=== MISSION SEARCH: timed out ===')
        return None

    # --- full mission -------------------------------------------------------
    def run_mission(self):
        log = self.get_logger()
        log.info('=== MISSION: START ===')

        if not self.wait_for_localization():
            return

        det = self.search_via_patrol()
        if det is None:
            log.error('Box never found -- mission aborted.')
            return

        # The patrol detection was taken while driving; let the base settle
        # (the patrol was just canceled) and take a fresh stationary front-cam
        # reading so the map-frame approach goal is accurate.
        self._stop_base()
        time.sleep(2.0)
        fresh = self.detect_box_front(timeout_sec=2.0)
        if fresh is not None:
            det = fresh
        bx, by, _ = det

        # APPROACH: box -> map, Nav2 to ~APPROACH_DIST in front, then wrist servo
        box_map = self.box_in_map(bx, by)
        robot_map = self.robot_in_map()
        if box_map is None or robot_map is None:
            log.error('TF unavailable for approach goal -- aborting.')
            return
        log.info(f'[mission] box in map: ({box_map[0]:.2f},{box_map[1]:.2f})')
        if not self.navigate_to(self.compute_approach_goal(box_map, robot_map)):
            log.error('Approach navigation failed -- aborting.')
            return
        if not self.search_and_approach():
            log.error('Visual approach failed -- aborting.')
            return

        # PICK
        if not self.pick_up_box():
            log.error('Pick failed -- aborting.')
            return

        # DELIVER (carry the box to the delivery point)
        log.info('=== MISSION: delivering to drop-off ===')
        if not self.navigate_to(self.make_map_goal(*DELIVERY_POSE)):
            log.error('Delivery navigation failed -- placing where we are.')
        self.place_box_down()

        # PARK
        log.info('=== MISSION: parking ===')
        self.navigate_to(self.make_map_goal(*PARKING_POSE))
        log.info('=== MISSION: DONE ===')


def main():
    rclpy.init()
    node = Mission()
    ex = rclpy.executors.MultiThreadedExecutor(4)
    ex.add_node(node)

    def task():
        time.sleep(3.0)
        node.run_mission()

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
