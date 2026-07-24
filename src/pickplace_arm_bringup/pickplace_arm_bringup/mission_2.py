#!/usr/bin/env python3
"""Mission 2: sort 3 coloured boxes from a table onto 3 matching-coloured columns.

Layout (map frame; map origin = robot's spawn pose):
  * A low table holds 3 boxes in a line -- red, green, blue.
  * 3 columns elsewhere, same x, y spaced 0.3 m, heights 8/12/10 cm -- kept low
    so every placement sits well inside the arm's accurate reach (and so the
    post-placement retreat clears the just-placed box) -- each coloured to match
    its box (red/green/blue).

For each box i (in order): drive to the table, claw-pick box i by colour,
drive to column i, centre the BASE on it (front camera, colour blob -- same
mechanism as the box pick), and place the box on top of that column at its
known height. Then drive to a final location and stop.

The arm/wrist orientation never changes across the whole mission: it holds
the zdown ("gripper straight down") pose from pick through carry through
place. Column alignment is done entirely by driving the base, never by
reorienting the arm -- there is no wrist-tilt scan step (an AprilTag-based
version of this used to exist; it needed a one-off wrist tilt to see a tag
lying flat on the column top, which conflicts with keeping the arm/wrist
orientation fixed, so placement uses the column's own body colour instead
of a tag).

Reuses Mission/NavAndPick (navigate_to, claw_approach/claw_pick, grab_below,
_face_box) and PickAndPlace (colour detect, gripper, move_pose).
"""
import math
import time
import threading

import rclpy
from geometry_msgs.msg import Twist
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters

from pickplace_arm_bringup.mission import Mission
from pickplace_arm_bringup.pick_and_place import (
    HOME_CONFIG, GRIPPER_X, GRIPPER_Y, GRIP_OPEN, BOX_ID, BOX_SIZE, GRASP_LINK,
    FINGER_LINKS, zdown_quat)
from pickplace_arm_bringup.search_and_pick import (
    APPROACH_LINEAR_GAIN, APPROACH_LINEAR_MAX, APPROACH_LINEAR_MIN,
    APPROACH_ANGULAR_GAIN, APPROACH_ANGULAR_MAX)

# --- layout (map frame) -------------------------------------------------------
TABLE_APPROACH = (1.45, 0.0, 0.0)      # pose the robot drives to before picking
TABLE_Z = 0.10                          # table-top height
TABLE_GRASP_Z = 0.13                    # gripper_base z to grasp a box on the table
# (There is no TABLE_X_OFFSET any more: the front camera's forward "bias" was
# never a calibration constant, it is the near-face-vs-centre geometry, and
# detect_box_front(depth=...) now resolves it from the blob's own extent for
# the ground box and the table box alike -- see pick_and_place._detect.)
BOXES = [                               # (colour, box map position) in pick order
    ('red',   (2.30, -0.16)),
    ('green', (2.30,  0.00)),
    ('blue',  (2.30,  0.16)),
]
COLUMNS = [                             # (column id, height, column map x,y)
    (0, 0.08, (-1.0, -0.30)),  # red   -> column 1,  8 cm
    (1, 0.12, (-1.0,  0.00)),  # green -> column 2, 12 cm
    # Column 3 was 16 cm, but at that height the post-placement retreat (capped
    # at OVER_Z_CEILING=0.21 by arm reach) cleared the just-placed box top
    # (0.16 + 0.045 = 0.205) by only ~0.5 cm, so the gripper clipped the box on
    # the way up. Lowered to 10 cm (box top 0.145, ~6.5 cm clearance -- the same
    # margin the two shorter columns already have). The model.sdf for
    # apriltag_column_3 was shortened to match.
    (2, 0.10, (-1.0,  0.30)),  # blue  -> column 3, 10 cm
]
FINAL_POSE = (0.0, -1.8, 0.0)

# Nav2 gets the base to a rough standoff in front of the column; from there
# the base is FINE-centred with the front camera (same colour-blob detector
# used for the table box, pointed at the column body instead), exactly like
# claw_approach centres the base under the gripper for a pick. The arm/wrist
# never moves for this -- it holds zdown throughout -- so there is no ARM
# reachability cliff to guard against (unlike the old AprilTag design, which
# read an actual measured position that could land beyond accurate arm
# reach): the visual servo itself drives to the right standoff distance.
# There IS still a camera-VISIBILITY cliff, though: if Nav2 arrives too far
# off-heading, the column is outside the front camera's FOV and
# approach_column's bounded _face_box correction (capped ~57 deg) cannot
# recover it -- confirmed live (column 1 arrival >57 deg off, zero pixels
# detected, approach aborted). Same reason the old AprilTag design tightened
# Nav2's arrival heading for this goal; still needed here for the same reason.
NAV_STANDOFF = 0.42     # Nav2 stops this far in front of the column
COLUMN_SIZE = 0.12      # column footprint (see models/apriltag_column_*/model.sdf)
# Stop the approach when the column's CENTRE reads this far ahead -- i.e. right
# under the gripper-down pose, exactly as CLAW_STOP_X does for the box pick.
# approach_column reads the column with depth=COLUMN_SIZE, so the reading is the
# centre of the column, not the centre of the face pointing at the camera (the
# front camera only ever sees that near face, 0.06 m closer).
#
# This used to be GRIPPER_X - 0.02 against the raw FACE reading, i.e. a centre
# at 0.42 -- which is NAV_STANDOFF, so the stop condition was already true the
# moment Nav2 arrived and the visual servo never actually corrected x at all;
# placement inherited Nav2's x error whole. Worse, the arm then had to reach
# 0.42 but was capped at GRIPPER_X + 0.03 = 0.41, so it always aimed ~1 cm
# short of the column's centre -- off the middle of the AprilTag, toward the
# near edge. Stopping at the centre instead puts the column under the arm's
# nominal 0.38 reach, so nothing is clamped and the servo does real work.
COLUMN_STOP_X = GRIPPER_X
# The box is released (not pinched) onto the column, so nothing re-centres it in
# y the way the closing jaws re-centre a grabbed box -- the placement y is only
# as good as how well the base centred on the column. Tightened from 0.02 to
# 0.012 so the drop lands on the middle of the AprilTag. (The arm itself always
# places on its y=0 centreline; see place_on_column -- it never yaws to chase a
# measured column y, which would rotate the arm.)
COLUMN_Y_TOL = 0.012

# nav2_params.yaml deliberately loosens yaw_goal_tolerance to 0.5 rad (~29deg)
# to stop the skid-steer oscillating (and tripping its own Spin recovery) at a
# tight tolerance -- but that is too loose for the column approach: an arrival
# heading error beyond approach_column's ~57deg correction cap leaves the
# column outside the front camera's view entirely. Tightened only for the
# column-approach goal, restored right after, so the looser default (which
# avoids the oscillation) still applies everywhere else (table approach,
# parking).
COLUMN_YAW_TOLERANCE = 0.20
DEFAULT_YAW_TOLERANCE = 0.5


class Mission2(Mission):
    def __init__(self):
        super().__init__()
        self._set_params_client = self.create_client(
            SetParameters, '/controller_server/set_parameters')
        self.get_logger().info('Mission 2 node ready')

    # --- Nav2 tuning ----------------------------------------------------------
    def _set_yaw_goal_tolerance(self, value, timeout_sec=3.0):
        """Set controller_server's goal-checker yaw tolerance at runtime.
        Returns True on success (best-effort: a failure just leaves the
        previous tolerance in place, which is still a valid, working value)."""
        log = self.get_logger()
        if not self._set_params_client.wait_for_service(timeout_sec=timeout_sec):
            log.warn('[nav] controller_server set_parameters service unavailable')
            return False
        req = SetParameters.Request()
        p = Parameter()
        p.name = 'general_goal_checker.yaw_goal_tolerance'
        p.value = ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=value)
        req.parameters = [p]
        future = self._set_params_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_sec)
        ok = future.done() and future.result() is not None and future.result().results[0].successful
        if not ok:
            log.warn(f'[nav] failed to set yaw_goal_tolerance={value}')
        return ok

    def approach_column(self, col_xy, color, timeout_sec=60.0):
        """Front-camera visual servo that drives the BASE until the `color`
        column sits directly under the gripper's fixed action point --
        mirrors claw_approach for the box pick exactly, just pointed at the
        column's body colour instead of the box. The arm is not touched here
        at all (it holds zdown, carrying the box, the whole time), so unlike
        the old AprilTag scan there is no wrist reorientation and no arm-
        reachability cliff to worry about: centring the BASE on the column is
        what puts it within reach, at a known, fixed local point."""
        log = self.get_logger()
        log.info(f'=== PLACE APPROACH: column (front-cam, base-only, {color}) ===')
        if col_xy is not None:
            self._face_box(col_xy)
        deadline = time.time() + timeout_sec
        twist = Twist()
        lost = 0
        while time.time() < deadline:
            det = self.detect_box_front(timeout_sec=0.25, color=color,
                                        depth=COLUMN_SIZE)
            if det is not None:
                lost = 0
                bx, by, _ = det
                if bx <= COLUMN_STOP_X and abs(by) <= COLUMN_Y_TOL:
                    self._stop_base()
                    log.info(f'[place] column centred (front {bx:.2f},{by:+.2f})')
                    return True
                fwd = max(0.0, bx - COLUMN_STOP_X)
                twist.linear.x = min(APPROACH_LINEAR_MAX,
                                     max(APPROACH_LINEAR_MIN, APPROACH_LINEAR_GAIN * fwd))
                twist.angular.z = max(-APPROACH_ANGULAR_MAX,
                                      min(APPROACH_ANGULAR_MAX,
                                          APPROACH_ANGULAR_GAIN * math.atan2(by, bx)))
            else:
                lost += 1
                if lost > 12:
                    self._stop_base()
                    log.warn('[place] lost the column -- aborting approach')
                    return False
                twist.linear.x *= 0.4
                twist.angular.z = 0.0
            for _ in range(4):
                self.cmd_vel_pub.publish(twist)
                time.sleep(0.03)
        self._stop_base()
        log.warn('[place] column approach timed out')
        return False

    # --- placement ----------------------------------------------------------
    def place_on_column(self, tag_id, height, col_xy, color):
        """Centre the base on the `color` column (front camera, base motion
        only -- the arm/wrist orientation never changes), then lower the held
        box straight down onto its known top height and release. Returns True
        on success."""
        log = self.get_logger()
        if not self.approach_column(col_xy, color):
            log.warn(f'[place] failed to centre on column {tag_id}')
            return False
        # Take a FRESH reading after the base settles rather than trusting
        # approach_column's last (pre-stop) reading: the skid-steer base has
        # momentum and keeps drifting a little after the stop command, so by
        # the time the arm would act, the column is not quite where it was
        # measured mid-approach -- this silently placed the box short/long of
        # the column top once (landed on the floor beside it, PLACE: DONE
        # notwithstanding). Same fix pattern as grab_below's fresh read before
        # descending onto a box.
        time.sleep(0.4)
        det = self.detect_box_front(timeout_sec=1.5, color=color,
                                    depth=COLUMN_SIZE)
        if det is None:
            log.warn(f'[place] lost sight of column {tag_id} after settling')
            return False
        # depth=COLUMN_SIZE makes det[0] the column's CENTRE -- which is where
        # the AprilTag is centred too -- so the box is lowered onto the middle of
        # the tag rather than somewhere on the near half of the column top. The
        # cap is only a reach guard now; after the approach above the centre
        # reads ~GRIPPER_X, so it should never bite.
        px = min(GRIPPER_X + 0.03, det[0])
        # Place on the arm's fixed y=0 centreline, NOT the measured column y:
        # reaching a sideways y would rotate the shoulder-yaw joint j1, and the
        # arm must never rotate. The base has already centred on the column in y
        # (approach_column, |by| <= COLUMN_Y_TOL), so y=0 already sits on the tag
        # centre; det[1] is used only to sanity-check that below.
        py = GRIPPER_Y
        if abs(det[1]) > COLUMN_Y_TOL + 0.01:
            log.warn(f'[place] column {tag_id} y={det[1]:+.3f} still off the '
                     f'gripper centreline after approach -- placing on centreline '
                     f'anyway (no arm yaw); drop may be off-centre in y')
        top_z = height + 0.03            # gripper_base z: box bottom rests on top
        # Cap the "over" clearance to what's actually reachable from the CARRY
        # branch at this x: a live /compute_ik sweep at x~0.41 (where px used to
        # be clamped; it is ~0.38 now, slightly closer in and so no worse for
        # reach) found the same-branch ceiling is z~0.22 (0.22 reachable, 0.23
        # is not, tested 3x) -- higher needs a branch flip, same failure mode as
        # the CARRY_POSITION bug. The strict carry->over-column move for the
        # 12cm column (over_z = 0.15+0.08 = 0.23) hit exactly this and failed
        # with NO_IK_SOLUTION, aborting with the box still held. 0.21 keeps a
        # safety margin below the measured 0.22 edge; the full 0.08 m
        # clearance still applies whenever it fits under that ceiling (true
        # for the two shorter columns).
        OVER_Z_CEILING = 0.21
        over_z = min(top_z + 0.08, OVER_Z_CEILING)
        log.info(f'=== PLACE: box onto column {tag_id} at ({px:.2f},{py:+.2f}) h={height} ===')
        # Check every move here: a failed move that goes unchecked leaves the
        # arm wherever it happened to stop, and detaching/releasing at THAT
        # unintended spot drops the box off-target while still logging
        # "PLACE: DONE" -- this is exactly what silently dropped the box on
        # the floor next to the column despite a good tag read. If either move
        # fails, abort with the box still attached/grasped (never detach or
        # open the gripper on an unconfirmed position).
        # strict: a failed direct move must never fall back to an unseeded
        # pose plan while holding the box -- that can pick any IK solution,
        # including one that swings the joints all the way around.
        if not self.move_pose(px, py, over_z, 0.0, label='over-column',
                              quat_xyzw=zdown_quat(0.0), strict=True):
            log.warn(f'[place] over-column move failed for column {tag_id} -- '
                     f'aborting placement (box still held)')
            return False
        if not self.move_pose(px, py, top_z, 0.0, cartesian=True,
                              label='lower-onto-column', quat_xyzw=zdown_quat(0.0)):
            log.warn(f'[place] lower-onto-column move failed for column {tag_id} '
                     f'-- aborting placement (box still held)')
            return False
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
            if not self.claw_pick(box_map, color=color, grasp_z=TABLE_GRASP_Z):
                log.error(f'Failed to pick the {color} box -- aborting.')
                return

            # back straight off the table so Nav2 can turn/plan without the table
            # (right in front of the robot) tripping its collision check.
            log.info('[mission2] backing off the table')
            self._drive_blind(-0.18, 3.5)
            self._stop_base()

            # 2) drive to a standoff in front of the column (facing it, yaw=pi
            # since the robot approaches from the +x table side), then visually
            # centre the base on it (front camera) and place the box on top.
            # Tighten Nav2's own arrival heading just for this goal:
            # approach_column's heading correction (_face_box) is capped at
            # ~57deg so it can't recover an arbitrarily bad arrival, and a
            # heading error beyond that puts the column outside the front
            # camera's view entirely (confirmed live). Restored right after so
            # the looser default (which avoids skid-steer oscillation) still
            # applies to every other goal.
            approach = (col_xy[0] + NAV_STANDOFF, col_xy[1], math.pi)
            self._set_yaw_goal_tolerance(COLUMN_YAW_TOLERANCE)
            nav_ok = self.navigate_to(self.make_map_goal(*approach))
            self._set_yaw_goal_tolerance(DEFAULT_YAW_TOLERANCE)
            if not nav_ok:
                log.error(f'Column {tag_id} navigation failed -- aborting.')
                return
            if not self.place_on_column(tag_id, height, col_xy, color):
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
