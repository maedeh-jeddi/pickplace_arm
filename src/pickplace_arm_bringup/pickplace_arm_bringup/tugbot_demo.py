#!/usr/bin/env python3
"""Minimal one-box pick-and-place demo for the Tugbot warehouse world.

No Nav2/AMCL: the table and column are both spawned within a couple of
metres of the robot's spawn point (inside the world's yellow hazard-taped
square), so instead of navigating there the base just turns in place to
face each one, then the existing front-camera visual-servo pick/place
(claw_pick / place_on_column) takes over exactly as it does mid-mission in
mission_2.py -- those never needed the map frame for anything but the
optional pre-turn (_face_box), which just no-ops without a map->base_link
TF. The turn itself is closed-loop on /imu yaw (search_and_pick's
_rotate_step is open-loop timing, tuned for small recovery sweeps, and
measured to under/over-shoot a full 180 deg turn badly enough on this
world's floor friction to lose the table entirely)."""
import math
import threading
import time

import rclpy
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Twist

from pickplace_arm_bringup.mission_2 import Mission2

TURN_ANGULAR_MAX = 0.4
TURN_TOLERANCE = 0.03
TURN_TIMEOUT_SEC = 20.0

# This downloaded warehouse world is full of same-coloured clutter (pallet
# labels, hazard markings, shelf trim) that the plain HSV blob detector --
# tuned for the clean/curated pickplace worlds -- picks up as false
# positives (e.g. a "red" hit 3+ m out and >1 m up, nowhere near the table).
# Gate detections to where the box/column actually are: ahead, close, and
# near table/gripper height. See mission.py's `_detect` docstring for the
# same technique already used for the Ionic world's column detection.
#
# The z bounds are relative to the LENS, so they moved +0.109 with it when the
# front camera was seated on the chassis front panel (FRONT_CAM_Z 0.332 ->
# 0.223); the WORLD slice admitted, 0.18..0.58 m off the floor, is unchanged.
DEMO_GATE = (0.05, 2.0, -0.6, 0.6, -0.04, 0.36)


class TugbotDemo(Mission2):
    COLUMN_DETECT_GATE = DEMO_GATE

    def __init__(self):
        super().__init__()
        self._imu_yaw = None
        self.create_subscription(Imu, '/imu', self._imu_cb, 10)

    def detect_box_front(self, timeout_sec=2.0, debug_save=False, color='blue', gate=None):
        if gate is None:
            gate = DEMO_GATE
        return super().detect_box_front(timeout_sec, debug_save, color, gate)

    def _imu_cb(self, msg):
        q = msg.orientation
        self._imu_yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                                   1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    def _turn_by(self, delta_yaw, timeout_sec=TURN_TIMEOUT_SEC):
        """Closed-loop in-place turn by `delta_yaw` radians using /imu."""
        log = self.get_logger()
        deadline = time.time() + 3.0
        while self._imu_yaw is None and time.time() < deadline:
            time.sleep(0.1)
        if self._imu_yaw is None:
            log.warn('[turn] no IMU data -- skipping turn')
            return
        target = self._imu_yaw + delta_yaw
        twist = Twist()
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            err = math.atan2(math.sin(target - self._imu_yaw),
                             math.cos(target - self._imu_yaw))
            if abs(err) < TURN_TOLERANCE:
                break
            twist.angular.z = max(-TURN_ANGULAR_MAX, min(TURN_ANGULAR_MAX, 1.5 * err))
            self.cmd_vel_pub.publish(twist)
            time.sleep(0.05)
        self._stop_base()
        log.info(f'[turn] done (target {math.degrees(delta_yaw):.0f} deg, '
                 f'err {math.degrees(err):.1f} deg)')
        time.sleep(0.6)

    def run_demo(self):
        log = self.get_logger()
        log.info('=== TUGBOT DEMO: START ===')

        # Face the table (behind spawn -- see tugbot_warehouse_view.launch.py).
        self._turn_by(math.pi)
        if not self.claw_pick(None, color='red',
                              grasp_z=self.LAYOUT_TABLE_GRASP_Z,
                              x_offset=self.LAYOUT_TABLE_X_OFFSET):
            log.error('Failed to pick the red box -- aborting.')
            return
        log.info('[mission2] backing off the table')
        self._drive_blind(-0.18, 3.5)
        self._stop_base()

        # Face the column (in front of spawn).
        self._turn_by(math.pi)
        if not self.place_on_column(0, 0.08, None, 'red'):
            log.error('Failed to place on the column -- aborting.')
            return

        log.info('=== TUGBOT DEMO: DONE ===')


def main():
    rclpy.init()
    node = TugbotDemo()
    ex = rclpy.executors.MultiThreadedExecutor(4)
    ex.add_node(node)

    def task():
        time.sleep(3.0)
        node.run_demo()

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
