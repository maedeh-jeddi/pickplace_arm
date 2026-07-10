#!/usr/bin/env python3
"""
Autonomous search-and-pick: drives the mobile base to find the box using the
wrist-mounted camera, then hands off to the existing, already-verified
pick-and-place sequence in pick_and_place.py (unmodified) for the precise
grasp.

Behavior:
  1) Arm holds a forward-and-down "search" pose so the camera can see ahead
     of the robot while it drives.
  2) SEARCH: spin in place, running the same HSV+point-cloud box detector
     used for the close-up pick, until the box is seen. If a full rotation
     finds nothing, drive forward a short distance and spin again.
  3) APPROACH: once seen, steer toward the box (proportional control on
     bearing + distance, both computed directly from detect_box_pose's
     base_link-frame result) until it's within the arm's known reach.
  4) Stop the base, then call the inherited run() -- the exact same
     precise scan+detect+grasp+place sequence pick_and_place.py already
     uses for a stationary robot, untouched.
"""
import math
import time
import threading

import rclpy
from geometry_msgs.msg import Twist

from pickplace_arm_bringup.pick_and_place import (
    PickAndPlace, scan_quat, SCAN_POSITION, SCAN_PITCH)

# --- forward-search vantage point (base_link frame) ---------------------------
# Shallower pitch than the close-up SCAN_POSITION/SCAN_PITCH in
# pick_and_place.py, so the camera can see further ahead while driving
# instead of straight down at a target assumed to already be close.
# Verified empirically to reliably detect/track a box from ~1.0m down to
# ~0.85m, but NOT closer -- a fixed downward-tilted camera has a near-range
# blind zone below that. Below TRANSITION_DISTANCE we switch to the already
# -proven close-up SCAN_POSITION/SCAN_PITCH (same pose pick_and_place.py's
# own precise scan uses, verified working down to 0.43m) for the final
# approach, instead of trying to tune a single pitch to cover both far and
# near range at once.
FORWARD_SEARCH_POSITION = (0.35, 0.00, 0.45)
FORWARD_SEARCH_PITCH = math.radians(-9.0)
TRANSITION_DISTANCE = 0.9        # m; switch to the close-up pose below this

# --- search state machine tuning ------------------------------------------------
SPIN_ANGULAR = 0.3              # rad/s while searching in place
SPIN_FULL_ROTATION_SEC = 22.0   # >= 2*pi/SPIN_ANGULAR, with margin
BLIND_FORWARD_LINEAR = 0.2      # m/s, when a full spin finds nothing
BLIND_FORWARD_SEC = 2.0

APPROACH_LINEAR_GAIN = 0.3
APPROACH_LINEAR_MAX = 0.25      # m/s
APPROACH_ANGULAR_GAIN = 1.2
APPROACH_ANGULAR_MAX = 0.6      # rad/s
STOP_DISTANCE = 0.45            # m; within the range pick_and_place.py's own
                                 # precise scan pose has been proven to
                                 # handle (verified at 0.43m and 0.60m).

SEARCH_TIMEOUT_SEC = 180.0


class SearchAndPick(PickAndPlace):
    def __init__(self):
        super().__init__()
        self.cmd_vel_pub = self.create_publisher(
            Twist, '/diff_drive_controller/cmd_vel_unstamped', 10)
        self.get_logger().info('Search-and-pick node ready')

    def _stop_base(self):
        self.cmd_vel_pub.publish(Twist())

    def _drive_blind(self, linear_x, duration_sec):
        twist = Twist()
        twist.linear.x = linear_x
        end = time.time() + duration_sec
        while time.time() < end:
            self.cmd_vel_pub.publish(twist)
            time.sleep(0.1)
        self._stop_base()

    def search_and_approach(self, timeout_sec=SEARCH_TIMEOUT_SEC):
        log = self.get_logger()
        log.info('=== SEARCH: START ===')

        sx, sy, sz = FORWARD_SEARCH_POSITION
        self.move_pose(sx, sy, sz, label='forward-search',
                       quat_xyzw=scan_quat(FORWARD_SEARCH_PITCH))
        close_range = False

        twist = Twist()
        deadline = time.time() + timeout_sec
        spin_deadline = time.time() + SPIN_FULL_ROTATION_SEC

        while time.time() < deadline:
            detection = self.detect_box_pose(timeout_sec=1.0)

            if detection is not None:
                bx, by, _bz = detection
                dist = math.hypot(bx, by)
                bearing = math.atan2(by, bx)
                log.info(f'[search] box seen: dist={dist:.2f}m '
                          f'bearing={math.degrees(bearing):.1f}deg')

                if dist < STOP_DISTANCE:
                    self._stop_base()
                    log.info('=== SEARCH: box within reach, stopping ===')
                    return True

                if not close_range and dist < TRANSITION_DISTANCE:
                    log.info('[search] within transition range -- switching '
                              'to the proven close-up scan pose')
                    self._stop_base()
                    cx, cy, cz = SCAN_POSITION
                    self.move_pose(cx, cy, cz, label='close-range',
                                   quat_xyzw=scan_quat(SCAN_PITCH))
                    close_range = True
                    spin_deadline = time.time() + SPIN_FULL_ROTATION_SEC
                    continue

                twist.linear.x = min(APPROACH_LINEAR_MAX,
                                      APPROACH_LINEAR_GAIN * dist)
                twist.angular.z = max(-APPROACH_ANGULAR_MAX, min(
                    APPROACH_ANGULAR_MAX, APPROACH_ANGULAR_GAIN * bearing))
                self.cmd_vel_pub.publish(twist)
                spin_deadline = time.time() + SPIN_FULL_ROTATION_SEC
            else:
                twist.linear.x = 0.0
                twist.angular.z = SPIN_ANGULAR
                self.cmd_vel_pub.publish(twist)
                if time.time() > spin_deadline:
                    log.info('[search] full rotation, nothing found -- '
                              'driving forward and retrying')
                    self._stop_base()
                    self._drive_blind(BLIND_FORWARD_LINEAR, BLIND_FORWARD_SEC)
                    spin_deadline = time.time() + SPIN_FULL_ROTATION_SEC

        self._stop_base()
        log.error('=== SEARCH: timed out without finding the box ===')
        return False


def main():
    rclpy.init()
    node = SearchAndPick()
    ex = rclpy.executors.MultiThreadedExecutor(4)
    ex.add_node(node)

    def task():
        time.sleep(3.0)
        if node.search_and_approach():
            node.run()
        else:
            node.get_logger().error('Search failed -- not attempting pick.')

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
