#!/usr/bin/env python3
"""Mission 2: sort 3 coloured boxes from a table onto 3 matching-coloured columns.

Layout (map frame; map origin = robot's spawn pose):
  * A low table holds 3 boxes in a line -- red, green, blue.
  * 3 columns elsewhere, same x, y spaced 0.3 m, heights 8/12/16 cm -- kept low
    so every placement sits well inside the arm's accurate reach -- each
    coloured to match its box (red/green/blue).

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
    HOME_CONFIG, GRIPPER_X, GRIP_OPEN, BOX_ID, BOX_SIZE, GRASP_LINK,
    FINGER_LINKS, zdown_quat)
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
COLUMNS = [                             # (column id, height, column map x,y)
    (0, 0.08, (-1.0, -0.30)),  # red   -> column 1,  8 cm
    (1, 0.12, (-1.0,  0.00)),  # green -> column 2, 12 cm
    (2, 0.16, (-1.0,  0.30)),  # blue  -> column 3, 16 cm
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
# Stop the approach when the column's front-camera reading reaches this x
# (mirrors CLAW_STOP_X for the box pick). The column is a 0.12 m cube, so the
# front camera -- which only sees its near face -- reads its centre as
# COLUMN_X_OFFSET further away than that face. Initial guess (0.06, half the
# column depth) measured a consistent ~2.2 cm overshoot across all 3 columns
# (compared the placed box's actual Gazebo pose to the column's known map
# x,y): red -0.022, green -0.023, blue -0.020 -- essentially identical, so a
# systematic bias, not noise. Reduced by that average (~0.022) to correct it.
COLUMN_STOP_X = GRIPPER_X - 0.02
COLUMN_Y_TOL = 0.02
COLUMN_X_OFFSET = 0.038

# nav2_params.yaml deliberately loosens yaw_goal_tolerance to 0.5 rad (~29deg)
# to stop the skid-steer oscillating (and tripping its own Spin recovery) at a
# tight tolerance -- but that is too loose for the column approach: an arrival
# heading error beyond approach_column's ~57deg correction cap leaves the
# column outside the front camera's view entirely. Also needed for the TABLE
# approach: measured (ground-truth Gazebo pose vs the front camera's own
# detection at grasp time) a real ~3.5cm lateral grasp-centering error that
# traced back to the base arriving at the table ~10.5deg off the intended
# straight-on heading -- an off-axis camera view biases the HSV-mask
# centroid (the box's visible faces shift with viewing angle), and unlike
# the column-placement overshoot (a FIXED bias, fixable with a constant
# offset) this error scales with arrival angle, so a fixed TABLE_X_OFFSET-
# style correction is the wrong tool; keeping the approach angle small is.
# Tightened for both the table and column approach goals, restored right
# after each, so the looser default (which avoids the oscillation) still
# applies everywhere else (parking, initial patrol).
TIGHT_YAW_TOLERANCE = 0.20
DEFAULT_YAW_TOLERANCE = 0.5


class Mission2(Mission):
    # --- layout (map frame) ---------------------------------------------------
    # These default to the warehouse layout (module constants above). A
    # subclass relocates the SAME mission into a different world just by
    # overriding them -- run_mission_2 below is layout-agnostic and reads only
    # these attributes, never the module globals. (NAV_STANDOFF and the
    # yaw-tolerance constants are tuning, not layout, so they stay shared.)
    LAYOUT_TABLE_APPROACH = TABLE_APPROACH
    LAYOUT_TABLE_GRASP_Z = TABLE_GRASP_Z
    LAYOUT_TABLE_X_OFFSET = TABLE_X_OFFSET
    LAYOUT_BOXES = BOXES
    LAYOUT_COLUMNS = COLUMNS
    LAYOUT_FINAL_POSE = FINAL_POSE
    # Optional camera-frame gate (xmin,xmax,ymin,ymax,zmin,zmax) for the column
    # colour detection (see PickAndPlace._detect). None in the empty warehouse
    # (no clutter to reject); a subclass sets it for a colourful world.
    COLUMN_DETECT_GATE = None
    # How close (front-cam forward reading, m) the base stops from the column
    # before placing, and the max over-column clearance height (m). Defaults
    # tuned for the warehouse; see place_on_column for how they interact with
    # arm reach. A subclass placing on TALL columns stops closer (smaller
    # placement x) so the arm can reach a higher over_z and clear the top.
    COLUMN_STOP_X = GRIPPER_X - 0.02
    OVER_Z_CEILING = 0.21
    # Columns at/above TALL_COLUMN_H are approached CLOSER (and may use a
    # higher clearance ceiling), because placing high needs reach the arm only
    # has when it is not extended far out. Defaults keep the warehouse's single
    # behaviour (threshold above every column height = never triggers).
    TALL_COLUMN_H = 99.0
    COLUMN_STOP_X_TALL = GRIPPER_X - 0.02
    OVER_Z_CEILING_TALL = 0.21

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

    def _stop_x_for(self, height):
        """Front-cam stop distance for a column of this height. TALL columns
        need the base CLOSER: placing on them needs a higher over-column
        clearance, and the arm's reachable z from the carry branch shrinks as
        it reaches further out (z~0.21 is already marginal at the default
        ~0.38 placement x, which is why the tall column's over-column move
        intermittently failed NO_IK_SOLUTION). Short columns keep the default
        distance, which places them to <5 mm -- approaching them closer was
        measured to make placement WORSE (closer-range detection is less
        accurate), so this is deliberately applied only to tall ones."""
        if height is not None and height >= self.TALL_COLUMN_H:
            return self.COLUMN_STOP_X_TALL
        return self.COLUMN_STOP_X

    def approach_column(self, col_xy, color, timeout_sec=60.0, height=None):
        """Front-camera visual servo that drives the BASE until the `color`
        column sits directly under the gripper's fixed action point --
        mirrors claw_approach for the box pick exactly, just pointed at the
        column's body colour instead of the box. The arm is not touched here
        at all (it holds zdown, carrying the box, the whole time), so unlike
        the old AprilTag scan there is no wrist reorientation and no arm-
        reachability cliff to worry about: centring the BASE on the column is
        what puts it within reach, at a known, fixed local point."""
        log = self.get_logger()
        stop_x = self._stop_x_for(height)
        log.info(f'=== PLACE APPROACH: column (front-cam, base-only, {color}) ===')
        if col_xy is not None:
            self._face_box(col_xy)
        deadline = time.time() + timeout_sec
        twist = Twist()
        lost = 0
        while time.time() < deadline:
            det = self.detect_box_front(timeout_sec=0.25, color=color,
                                        gate=self.COLUMN_DETECT_GATE)
            if det is not None:
                lost = 0
                bx, by, _ = det
                if bx <= stop_x and abs(by) <= COLUMN_Y_TOL:
                    self._stop_base()
                    log.info(f'[place] column centred (front {bx:.2f},{by:+.2f})')
                    return True
                fwd = max(0.0, bx - stop_x)
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
        if not self.approach_column(col_xy, color, height=height):
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
        det = self.detect_box_front(timeout_sec=1.5, color=color)
        if det is None:
            log.warn(f'[place] lost sight of column {tag_id} after settling')
            return False
        px = min(GRIPPER_X + 0.03, det[0] + COLUMN_X_OFFSET)
        py = det[1]
        top_z = height + 0.03            # gripper_base z: box bottom rests on top
        # The over-column waypoint must clear the column TOP: the held box hangs
        # ~0.08 m below gripper_base, so its bottom is (over_z - 0.08). To pass
        # OVER the column during the horizontal approach that must exceed the
        # column height, i.e. over_z >= height + 0.08 (+ margin). But over_z is
        # also bounded by IK reach: from the CARRY branch the same-branch
        # z-ceiling depends on the reach x -- ~0.22 m at x~0.41, rising to ~0.27
        # at x<=0.34 (verified by /compute_ik sweep). The tallest column (0.16)
        # needs over_z ~0.24-0.27, unreachable at the default ~0.39 placement x
        # -- which is exactly why the box caught the column's near-top edge and
        # tipped onto the floor there. The fix pairs a CLOSER column stop (see
        # COLUMN_STOP_X: smaller placement x -> higher reachable z) with a
        # raised OVER_Z_CEILING; the min() still guards reach.
        ceiling = (self.OVER_Z_CEILING_TALL if height >= self.TALL_COLUMN_H
                   else self.OVER_Z_CEILING)
        over_z = min(top_z + 0.08, ceiling)
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

        for (color, box_xy), (tag_id, height, col_xy) in zip(self.LAYOUT_BOXES,
                                                             self.LAYOUT_COLUMNS):
            log.info(f'--- {color} box -> column {tag_id} (h={height}) ---')

            # 1) drive to the table and pick the coloured box off it. Tighten
            # Nav2's arrival heading here too (see TIGHT_YAW_TOLERANCE): an
            # off-axis arrival biases the front camera's HSV-centroid grasp
            # reading, which showed up as a real ~3.5cm lateral grasp-
            # centering error (confirmed via ground-truth Gazebo pose vs the
            # camera's own detection at grasp time) tracing back to a ~10.5deg
            # arrival heading error.
            self._set_yaw_goal_tolerance(TIGHT_YAW_TOLERANCE)
            nav_ok = self.navigate_to(self.make_map_goal(*self.LAYOUT_TABLE_APPROACH))
            self._set_yaw_goal_tolerance(DEFAULT_YAW_TOLERANCE)
            if not nav_ok:
                log.error('Table navigation failed -- aborting.')
                return
            box_map = (box_xy[0], box_xy[1])
            if not self.claw_pick(box_map, color=color,
                                  grasp_z=self.LAYOUT_TABLE_GRASP_Z,
                                  x_offset=self.LAYOUT_TABLE_X_OFFSET):
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
            self._set_yaw_goal_tolerance(TIGHT_YAW_TOLERANCE)
            nav_ok = self.navigate_to(self.make_map_goal(*approach))
            self._set_yaw_goal_tolerance(DEFAULT_YAW_TOLERANCE)
            if not nav_ok:
                log.error(f'Column {tag_id} navigation failed -- aborting.')
                return
            if not self.place_on_column(tag_id, height, col_xy, color):
                log.error(f'Failed to place on column {tag_id} -- aborting.')
                return

        log.info('=== MISSION 2: parking ===')
        self.navigate_to(self.make_map_goal(*self.LAYOUT_FINAL_POSE))
        log.info('=== MISSION 2: DONE ===')


# --- Ionic-world layout -------------------------------------------------------
# The Ionic restaurant world (worlds/ionic.sdf) is a bigger, cluttered interior;
# its map frame equals the Gazebo WORLD frame (the occupancy grid is generated
# from the world's own collision geometry, so map coordinates ARE world
# coordinates). The whole warehouse layout is translated by (+5.70, +1.5) into a
# verified-clear pocket around world (4.7..8.0, 1.5): the table sits at world
# (8.0, 1.5) and the columns 3.3 m away at world (4.7, 1.5) -- identical relative
# geometry to the warehouse, so only these coordinates change. Robot spawns at
# world (6.5, 1.5); see mission_2_ionic.launch.py / amcl_ionic.yaml. Positions
# were checked clear against the generated map before use.
class Mission2Ionic(Mission2):
    LAYOUT_TABLE_APPROACH = (7.15, 1.5, 0.0)
    LAYOUT_BOXES = [
        ('red',   (8.0, 1.34)),
        ('green', (8.0, 1.50)),
        ('blue',  (8.0, 1.66)),
    ]
    LAYOUT_COLUMNS = [
        (0, 0.08, (4.7, 1.20)),   # red   -> column 1,  8 cm
        (1, 0.12, (4.7, 1.50)),   # green -> column 2, 12 cm
        (2, 0.16, (4.7, 1.80)),   # blue  -> column 3, 16 cm
    ]
    LAYOUT_FINAL_POSE = (6.3, 1.5, 0.0)
    # The Ionic restaurant is decorated white + navy blue (walls, ceiling
    # beams, pillars) -- the same blue as the blue box/column -- so the raw
    # front-camera blue blob is dominated by architecture and the colour servo
    # locked onto a wall instead of the column (blue placement failed while
    # red/green succeeded). This camera-frame gate (X-forward, Y-left, Z-up,
    # metres) keeps only points DEAD AHEAD (x 0.05-0.9 m), roughly CENTRED
    # (|y| < 0.35 m, rejects the side walls) and LOW (z -0.2..0.2 m, rejects
    # the ceiling beams) -- i.e. where a column the robot has driven up to
    # actually is. Verified against the failing frame: the distractor centroid
    # sat at camera (0.76, 0.58, 0.24) -- outside both the y and z bounds.
    COLUMN_DETECT_GATE = (0.05, 0.9, -0.35, 0.35, -0.2, 0.2)
    # The two SHORTER columns keep the warehouse-default stop (0.36) and
    # clearance (0.21) -- they place to <5 mm here, and approaching them closer
    # measurably made them WORSE (closer-range detection is less accurate), so
    # that stays untouched. Only the TALL (16 cm) column is treated specially:
    # placing on it needs an over-column clearance of z~0.21, which at the
    # default ~0.38 placement x sits right at the arm's reachable edge from the
    # carry branch -- it worked in some runs and failed NO_IK_SOLUTION in
    # others. Stopping 6 cm closer for that column only (placement x ~0.34)
    # moves it well inside reach (/compute_ik from the carry seed: z up to 0.25
    # is comfortable at x 0.30-0.34), and allows a slightly higher ceiling for
    # real clearance over the 0.16 top (box bottom = over_z - 0.03).
    TALL_COLUMN_H = 0.14
    COLUMN_STOP_X_TALL = 0.30
    OVER_Z_CEILING_TALL = 0.23


def _run(node_cls):
    rclpy.init()
    node = node_cls()
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


def main():
    _run(Mission2)


def main_ionic():
    _run(Mission2Ionic)


if __name__ == '__main__':
    main()
