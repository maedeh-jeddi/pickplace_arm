#!/usr/bin/env python3
"""Mission 2: sort 3 coloured boxes from a table onto 3 AprilTag columns.

Layout (map frame; map origin = robot's spawn pose):
  * A low table holds 3 boxes in a line -- red, green, blue.
  * 3 columns elsewhere, same x, y spaced 0.3 m, heights 8/12/16 cm -- kept low
    so every placement sits well inside the arm's accurate reach -- each with a
    tag36h11 AprilTag (IDs 0/1/2) lying flat on its TOP surface, read by the
    wrist/gripper camera.

For each box i (in order): drive to the table, claw-pick box i by colour, drive
to column i, use its AprilTag to line up, and place the box on top of that
column. Then drive to a final location and stop.

Reuses Mission/NavAndPick (navigate_to, claw_approach/claw_pick, grab_below,
_face_box) and PickAndPlace (colour detect, gripper, move_pose). The AprilTag
pose comes from apriltag_ros via TF (front_camera_optical_link -> tag36h11:<id>),
which we look up in base_link.
"""
import math
import time
import threading

import rclpy
from geometry_msgs.msg import Twist
import tf2_ros

from pickplace_arm_bringup.mission import Mission
from pickplace_arm_bringup.pick_and_place import (
    HOME_CONFIG, GRIPPER_X, GRIP_OPEN, BOX_ID, BOX_SIZE, GRASP_LINK,
    FINGER_LINKS, zdown_quat, scan_quat)
from pickplace_arm_bringup.search_and_pick import (
    APPROACH_LINEAR_GAIN, APPROACH_LINEAR_MAX, APPROACH_LINEAR_MIN,
    APPROACH_ANGULAR_GAIN, APPROACH_ANGULAR_MAX)

# --- layout (map frame) -------------------------------------------------------
TABLE_APPROACH = (1.45, 0.0, 0.0)      # pose the robot drives to before picking
TABLE_Z = 0.10                          # table-top height
TABLE_GRASP_Z = 0.13                    # gripper_base z to grasp a box on the table
# The front camera's ~2 cm "reads short" bias was calibrated for a box on the
# GROUND and does not hold for one raised on the table: applying it there lands
# the jaws on the far edge of the 4.5 cm box and shoves it off instead of
# grasping (this is what kept knocking the green box away). Descend on the
# measured position instead.
TABLE_X_OFFSET = 0.0
BOXES = [                               # (colour, box map position) in pick order
    ('red',   (2.30, -0.16)),
    ('green', (2.30,  0.00)),
    ('blue',  (2.30,  0.16)),
]
COLUMNS = [                             # (tag id, height, column map x,y)
    (0, 0.08, (-1.0, -0.30)),  # red   -> column 1 (tag 0),  8 cm
    (1, 0.12, (-1.0,  0.00)),  # green -> column 2 (tag 1), 12 cm
    (2, 0.16, (-1.0,  0.30)),  # blue  -> column 3 (tag 2), 16 cm
]
FINAL_POSE = (0.0, -1.8, 0.0)

# The tags sit flat on the column TOPS. The robot drives (Nav2 + known column
# position) to a standoff where the column top is under the arm, then dips the
# WRIST/gripper camera over it in a raised, pitched-down scan pose to read the
# tag -- accurate to a cm or two, independent of AMCL drift -- and places the
# held box on the measured tag position.
NAV_STANDOFF = 0.42     # Nav2 stops this far in front of the column
# Furthest forward the arm places ACCURATELY. Measured: a column at 0.43 m lands
# dead centre, 0.46 m still lands on top, but by 0.47 m the arm is extended
# enough that the box lands ~0.1 m off. Anything past PLACE_REACH_OK triggers a
# Nav2 re-approach + fresh scan rather than a sloppy long-reach place.
PLACE_X_MAX = 0.45
PLACE_REACH_OK = 0.44
# Wrist-scan pose: reach forward + up and tilt the camera forward-down so it
# frames the column top ~0.4 m ahead (not the robot's own chassis). The camera
# itself adds ~28.6 deg of down-tilt, so this pitch stays shallow; the grasped
# box rides up-and-back (out of this forward-down view).
SCAN_X = 0.28
SCAN_Z = 0.38
SCAN_PITCH = math.radians(32.0)
TAG_FRAME = 'tag36h11:{}'               # apriltag_ros TF frame per id


class Mission2(Mission):
    def __init__(self):
        super().__init__()
        self.get_logger().info('Mission 2 node ready')

    # --- perception ---------------------------------------------------------
    def detect_tag(self, tag_id, timeout_sec=1.0, max_age_sec=1.0):
        """Column tag (x, y) in base_link from apriltag_ros TF, or None.

        Rejects STALE transforms: once the tag leaves the camera's view tf2 keeps
        returning the last transform forever, which silently looks like a live
        detection and makes the arm place at a position the robot has since
        driven away from."""
        frame = TAG_FRAME.format(tag_id)
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            try:
                tf = self.tf_buffer.lookup_transform(
                    'base_link', frame, rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.2))
                stamp = rclpy.time.Time.from_msg(tf.header.stamp)
                age = (self.get_clock().now() - stamp).nanoseconds / 1e9
                if age > max_age_sec:
                    time.sleep(0.05)
                    continue
                t = tf.transform.translation
                return (t.x, t.y)
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException):
                time.sleep(0.05)
        return None

    # --- frame geometry -----------------------------------------------------
    def _transform_point(self, target, source, x, y, timeout_sec=3.0):
        """Planar point (x, y) in `source` frame expressed in `target` frame,
        via the live TF. Returns (rx, ry) or None."""
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            try:
                tf = self.tf_buffer.lookup_transform(
                    target, source, rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.4))
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException):
                time.sleep(0.15)
                continue
            t = tf.transform.translation
            q = tf.transform.rotation
            yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                             1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            rx = math.cos(yaw) * x - math.sin(yaw) * y + t.x
            ry = math.sin(yaw) * x + math.cos(yaw) * y + t.y
            return (rx, ry)
        return None

    def column_in_base(self, col_xy, timeout_sec=3.0):
        """Known column map (x, y) expressed in base_link (AMCL map->base_link).
        Coarse (~0.1-0.2 m AMCL error) -- only used as a rough aim bearing."""
        return self._transform_point('base_link', 'map', col_xy[0], col_xy[1],
                                     timeout_sec)

    def _drive_distance(self, dist, speed=0.10):
        """Drive straight forward `dist` metres, closed-loop on odometry.

        A fixed-duration open-loop burst (_drive_blind) is unreliable for short
        hops: the base's velocity deadband/inertia eats a 1-2 s command and it
        barely moves. Here we watch odom and stop when the distance is covered.
        The deadline is capped just above the expected travel time so a failed
        odom lookup can never turn this into a long uncontrolled drive."""
        start = self._transform_point('odom', 'base_link', 0.0, 0.0)
        if start is None:
            self._drive_blind(speed if dist >= 0 else -speed, abs(dist) / speed)
            return
        twist = Twist()
        twist.linear.x = speed if dist >= 0 else -speed
        # Budget generously: with the Nav2 stack up, something else is also
        # driving this topic and the base covers only ~45% of the commanded
        # distance per unit time (measured). The odom check below is what
        # actually ends the drive, so a loose deadline is only a safety net.
        deadline = time.time() + abs(dist) / speed * 4.0 + 2.0
        while time.time() < deadline:
            cur = self._transform_point('odom', 'base_link', 0.0, 0.0, timeout_sec=1.0)
            if cur is not None:
                moved = math.hypot(cur[0] - start[0], cur[1] - start[1])
                if moved >= abs(dist):
                    break
            self.cmd_vel_pub.publish(twist)
            time.sleep(0.05)
        self._stop_base()

    def scan_column_top(self, tag_id, col_xy, tries=10):
        """Dip the wrist/gripper camera over the column in a raised, pitched-down
        scan pose and read the top AprilTag. Returns the column-top centre
        (tx, ty) in base_link, or None. First nudges toward the known column
        bearing so the top sits ahead of the camera."""
        log = self.get_logger()
        log.info(f'=== PLACE: scan column {tag_id} top for its tag ===')
        # face the known column bearing (AMCL) so the top is straight ahead
        bp = self.column_in_base(col_xy)
        if bp is not None and abs(math.atan2(bp[1], bp[0])) > 0.10:
            self._rotate_step(max(-0.5, min(0.5, math.atan2(bp[1], bp[0]))))
        # close the distance (known column position) so the top sits under the
        # arm's reach before we scan/place -- Nav2/AMCL may leave it too far.
        bp = self.column_in_base(col_xy)
        if bp is not None and bp[0] > 0.42:
            drive = bp[0] - 0.38
            self._drive_blind(0.08, drive / 0.08)
            self._stop_base()
            time.sleep(0.3)
        # raise + pitch the wrist camera down over the column top
        self.move_pose(SCAN_X, 0.0, SCAN_Z, 0.0, label='column-top scan',
                       quat_xyzw=scan_quat(SCAN_PITCH))
        time.sleep(0.5)
        det = None
        for _ in range(tries):
            det = self.detect_tag(tag_id, timeout_sec=0.6)
            if det is not None:
                break
            time.sleep(0.3)
        if det is None:
            log.warn(f'[place] never saw top tag {tag_id}')
            return None
        tx, ty = det
        log.info(f'[place] top tag {tag_id} at ({tx:.2f},{ty:+.2f})')
        # Deliberately do NOT drive the base between reading the tag and placing.
        # Any base motion here invalidates the reading: once the tag leaves view
        # tf2 keeps composing the stale camera->tag with a live base->camera
        # chain, so the "re-read" looks fresh but is not, and the arm places
        # where the robot no longer is. The columns are low enough that the arm
        # reaches them from where Nav2 parks, so sense and place from one pose.
        return (tx, ty)

    # --- placement ----------------------------------------------------------
    def place_on_column(self, tag_id, height, col_xy):
        """Read the column's top tag with the wrist camera, then lower the held
        box straight down onto that spot and release. Returns True on success."""
        log = self.get_logger()
        top = self.scan_column_top(tag_id, col_xy)
        if top is None:
            return False
        # Nav2's goal tolerance sometimes parks us short, leaving the column
        # beyond the arm's reach; placing there would clamp to PLACE_X_MAX and
        # drop the box short of the top. Close in with Nav2 (raw cmd_vel doesn't
        # reliably move this base) and take a COMPLETELY fresh scan -- never
        # reuse the old reading across a base move.
        if top[0] > PLACE_REACH_OK:
            # Creep in on odometry (closed-loop: it drives until odom confirms
            # the distance, so it is immune to the base only covering ~45% of a
            # commanded burst while Nav2 is up). Nav2 itself refuses to plan a
            # goal this close to the column, so a re-navigate does not work.
            creep = top[0] - 0.40
            log.info(f'[place] column {tag_id} at {top[0]:.2f} m is beyond accurate '
                     f'reach -- creeping in {creep:.2f} m')
            self._drive_distance(creep)
            time.sleep(0.5)
            # Take a genuinely fresh scan afterwards. Never reuse the pre-creep
            # reading: it is measured from a pose the robot has since left.
            again = self.scan_column_top(tag_id, col_xy)
            if again is not None:
                top = again
                log.info(f'[place] after creep: ({top[0]:.2f},{top[1]:+.2f})')
        px = min(PLACE_X_MAX, top[0])
        py = top[1]
        top_z = height + 0.03            # gripper_base z: box bottom rests on top
        over_z = top_z + 0.08
        log.info(f'=== PLACE: box onto column {tag_id} at ({px:.2f},{py:+.2f}) h={height} ===')
        self.move_pose(px, py, over_z, 0.0, label='over-column',
                       quat_xyzw=zdown_quat(0.0))
        self.move_pose(px, py, top_z, 0.0, cartesian=True, label='lower-onto-column',
                       quat_xyzw=zdown_quat(0.0))
        self.arm.detach_collision_object(BOX_ID)
        time.sleep(0.3)
        self.gripper(GRIP_OPEN, 'release')
        self.move_pose(px, py, over_z, 0.0, cartesian=True, label='retreat',
                       quat_xyzw=zdown_quat(0.0))
        self.arm.remove_collision_object(BOX_ID)
        self.move_config(HOME_CONFIG, 'gripper-down ready')
        # back the base off the column so Nav2 can plan away (right up against
        # a solid column it can't find a valid start pose otherwise).
        log.info('[place] backing off the column')
        self._drive_blind(-0.22, 3.5)
        self._stop_base()
        log.info(f'=== PLACE: DONE (column {tag_id}) ===')
        return True

    # --- full mission -------------------------------------------------------
    def run_mission_2(self):
        log = self.get_logger()
        log.info('=== MISSION 2: START ===')
        if not self.wait_for_localization():
            return

        for (color, box_xy), (tag_id, height, col_xy) in zip(BOXES, COLUMNS):
            log.info(f'--- {color} box -> column {tag_id} (h={height}) ---')

            # 1) drive to the table and pick the coloured box off it
            if not self.navigate_to(self.make_map_goal(*TABLE_APPROACH)):
                log.error('Table navigation failed -- aborting.')
                return
            box_map = (box_xy[0], box_xy[1])
            if not self.claw_pick(box_map, color=color, grasp_z=TABLE_GRASP_Z,
                                  x_offset=TABLE_X_OFFSET):
                log.error(f'Failed to pick the {color} box -- aborting.')
                return

            # back straight off the table so Nav2 can turn/plan without the table
            # (right in front of the robot) tripping its collision check.
            log.info('[mission2] backing off the table')
            self._drive_blind(-0.18, 3.5)
            self._stop_base()

            # 2) drive to a standoff in front of the column (facing it, yaw=pi
            # since the robot approaches from the +x table side), then visually
            # servo onto its tag and place the box on the column top.
            approach = (col_xy[0] + NAV_STANDOFF, col_xy[1], math.pi)
            if not self.navigate_to(self.make_map_goal(*approach)):
                log.error(f'Column {tag_id} navigation failed -- aborting.')
                return
            if not self.place_on_column(tag_id, height, col_xy):
                log.error(f'Failed to place on column {tag_id} -- aborting.')
                return

        log.info('=== MISSION 2: parking ===')
        self.navigate_to(self.make_map_goal(*FINAL_POSE))
        log.info('=== MISSION 2: DONE ===')


def main():
    rclpy.init()
    node = Mission2()
    ex = rclpy.executors.MultiThreadedExecutor(4)
    ex.add_node(node)

    def task():
        time.sleep(3.0)
        node.run_mission_2()

    t = threading.Thread(target=task, daemon=True)
    t.start()
    try:
        # rclpy intermittently raises RCLError out of spin() ("wait set index
        # ... out of bounds") from an action-client race under load. Left
        # uncaught it kills the executor and the mission thread hangs forever,
        # so resume spinning instead of dying on it.
        while rclpy.ok():
            try:
                ex.spin()
                break
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                node.get_logger().warn(f'[executor] recovered from spin error: {exc}')
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
