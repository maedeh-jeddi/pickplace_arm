#!/usr/bin/env python3
"""
Full pick-and-place for the pickplace_arm in Gazebo Harmonic.

Uses MoveIt (via pymoveit2) for collision-aware motion planning of the arm,
the ros2_control gripper controller for the physical grasp, and the MoveIt
planning scene (attach/detach) so the box is carried correctly and shown in
RViz. The box is a real, physics-enabled model spawned in Gazebo, so a
successful run physically moves it from the pick pose to the place pose.

The box's (x, y) is not known in advance: the arm moves to a fixed scan pose,
the wrist-mounted RGB-D camera locates the box by color in the point cloud,
and that detected position (transformed into base_link via TF) drives the
grasp -- move the box and re-run and it is found and picked again.

Geometry (all arm poses are for the `gripper_base` link, in `base_link`):
  * pick  : box on the ground, (x, y) from detection -> grasp z 0.03, pre-grasp z 0.15
  * place : ground at ~(0.50, 0.35)
Grasp/place z were verified reachable with compute_ik (z-down gripper).
"""
import math
import time
import threading
from threading import Lock

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration as RclDuration
from sensor_msgs.msg import PointCloud2, JointState
from sensor_msgs_py import point_cloud2
from geometry_msgs.msg import PointStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

import tf2_ros
import tf2_geometry_msgs  # noqa: F401  (registers PointStamped transform support)

from pymoveit2 import MoveIt2

ARM_JOINTS = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
# Indices of the pure ROLL joints (rotate about their own axis, limits +/-pi):
# j1 (base yaw), j4 (forearm roll), j6 (wrist roll). For these, an angle and
# angle +/- 2*pi are the SAME orientation, so a goal near one +/-pi limit can be
# reached from near the other by commanding the equivalent value -- avoiding a
# useless ~2*pi unwind. (j2/j3/j5 are pitch joints with real sub-2*pi ranges.)
ROLL_JOINT_IDX = (0, 3, 5)
# Roll joints are limited to +/-2*pi in the URDF (a small margin below keeps the
# normalized equivalent safely inside the limit).
ROLL_LIMIT = 2.0 * math.pi - 0.02
GRIPPER_JOINTS = ['left_finger_joint', 'right_finger_joint']
GRASP_LINK = 'gripper_base'
FINGER_LINKS = ['left_finger', 'right_finger', 'gripper_base']

# --- task geometry (base_link frame) -----------------------------------------
BOX_ID = 'target_box'
BOX_SIZE = 0.045
# Re-derived via /compute_ik sweep after the arm links were shortened 30%
# (old reach ~0.94m -> new ~0.66m); (0.50, 0.35) is no longer reachable.
PLACE_XY = (0.30, 0.20)
GRASP_Z = 0.03          # gripper_base z when grasping a ground box
APPROACH_Z = 0.15       # pre-grasp / lift height
GRIP_OPEN = 0.03
GRIP_CLOSED = 0.0

# --- grasp verification -------------------------------------------------------
# After the jaws close, a finger joint held OPEN by the box reads clearly above
# an empty (fully-closed) grasp: empirically the finger position is ~0.000 when
# the jaws close on air, but 0.004-0.005 when a box (half-width 0.0225) is
# pinched between them (the box squeezes but the pads never reach 0). A box
# grasped nearer the arm's reach edge reads a bit lower (~0.0024), so the
# holding-vs-empty threshold is set with margin below the reliable held value.
FINGER_HELD_MIN = 0.0015
# Grasp attempts before giving up: a fresh scan + descend each time, so a box
# nudged by a missed first attempt is re-located and re-grasped instead of the
# robot silently carrying nothing.
MAX_GRASP_ATTEMPTS = 3

# Compact "carry" pose: box held low and centered over the base so it rides
# stably while the mobile base drives to the delivery point.
CARRY_POSITION = (0.26, 0.00, 0.18)

# Neutral / "ready" arm configuration (joint angles j1..j6): the gripper points
# STRAIGHT DOWN, reaching ~0.38 m ahead at ~0.22 m height -- a "claw" ready pose.
# The arm spawns in this pose (see initial_value in pickplace_arm.gazebo.xacro)
# and holds it through search + approach, so the mobile base only has to drive
# the box directly under the gripper (front camera guides that), then the gripper
# descends straight down onto it -- no wrist reorientation, no gripper spin.
# (This IS the reachable straight-down IK for gripper_base at (0.38,0,0.22).)
HOME_CONFIG = [0.0, 0.52, 1.14, -3.14, -1.48, 0.0]

# Claw geometry: where the gripper sits (base_link) in the ready pose, and the
# grasp/lift heights. The base positions the box under GRIPPER_X/Y, then the
# gripper descends straight down.
GRIPPER_X = 0.38
GRIPPER_Y = 0.0
READY_Z = 0.22
# The front camera reads the box's forward distance ~2 cm short of ground truth
# (measured); add this back when descending onto it.
FRONT_X_OFFSET = 0.02

# Expected box centroid height in base_link frame (ground plane, see add_box):
# used only as a sanity check against the detected z, not as the commanded z.
EXPECTED_BOX_Z = -0.05 + BOX_SIZE / 2.0

# --- perception ---------------------------------------------------------------
# Fixed vantage point the arm moves to before scanning for the box. Chosen so
# the eye-in-hand camera's frustum covers the reachable table area on the
# ground plane; verify/tune empirically (see detect_box_pose debug dump).
# Position re-derived via /compute_ik sweep after the arm links were
# shortened 30% (old reach ~0.94m -> new ~0.66m); (0.40, 0.60) is no longer
# reachable. Pitch is unchanged -- it's a property of the camera geometry,
# not the arm length.
SCAN_POSITION = (0.22, 0.00, 0.40)
SCAN_PITCH = math.radians(55.0)

# HSV bounds (OpenCV H 0-180) for each box colour. Red wraps around H=0, so it
# needs TWO ranges. Each entry is a list of (lower, upper) HSV tuples; a pixel
# matches the colour if it falls in ANY of the ranges.
COLOR_HSV = {
    'blue':  [((95, 120, 60), (115, 255, 255))],
    'green': [((35, 80, 40), (85, 255, 255))],
    'red':   [((0, 100, 50), (10, 255, 255)), ((170, 100, 50), (180, 255, 255))],
}
# Backward-compatible default (the original single blue box).
HSV_LOWER = COLOR_HSV['blue'][0][0]
HSV_UPPER = COLOR_HSV['blue'][0][1]
MIN_VALID_PIXELS = 30


def qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz)


def zdown_quat(yaw):
    """Gripper pointing straight down, yawed about world z (xyzw)."""
    cz, sz = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return qmul((0.0, 0.0, sz, cz), (1.0, 0.0, 0.0, 0.0))


def scan_quat(pitch, yaw=0.0):
    """Gripper tilted forward-down by `pitch` from horizontal, then yawed
    about world z (xyzw). Distinct from zdown_quat: at pitch=0 the gripper
    (and the camera mounted on it) points straight out horizontally."""
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    yaw_q = (0.0, 0.0, sy, cy)
    pitch_q = (0.0, sp, 0.0, cp)
    return qmul(yaw_q, pitch_q)


class PickAndPlace(Node):
    def __init__(self):
        super().__init__('pick_and_place')
        cbg = ReentrantCallbackGroup()

        self.arm = MoveIt2(
            node=self, joint_names=ARM_JOINTS, base_link_name='base_link',
            end_effector_name=GRASP_LINK, group_name='arm', callback_group=cbg)
        self.arm.max_velocity = 0.30
        self.arm.max_acceleration = 0.30

        # Scan pose used by run() to locate the box before grasping. Kept as
        # instance attributes so subclasses (e.g. the mobile search-and-pick)
        # can substitute a pose whose detection range matches where they stop
        # the base, without duplicating run().
        self.scan_position = SCAN_POSITION
        self.scan_pitch = SCAN_PITCH

        self.gripper_pub = self.create_publisher(
            JointTrajectory, '/gripper_controller/joint_trajectory', 10)

        # --- perception: point cloud subscriptions + TF ---
        # Wrist (eye-in-hand) RGB-D for the precise grasp scan, and the
        # base-mounted front RGB-D for detecting the box while driving.
        self._cloud_lock = Lock()
        self._latest_cloud = None
        self.create_subscription(
            PointCloud2, '/camera/points', self._cloud_cb, 1,
            callback_group=cbg)
        self._front_lock = Lock()
        self._front_cloud = None
        self.create_subscription(
            PointCloud2, '/front_camera/points', self._front_cloud_cb, 1,
            callback_group=cbg)

        # Latest joint positions -- used to VERIFY a grasp actually holds the
        # box (finger positions) and to seed/normalise IK so arm moves take the
        # nearest, simplest joint path (current arm config).
        self._joint_pos = {}
        self.create_subscription(
            JointState, '/joint_states', self._joint_state_cb, 10,
            callback_group=cbg)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.get_logger().info('Pick-and-place node ready')

    def _cloud_cb(self, msg):
        with self._cloud_lock:
            self._latest_cloud = msg

    def _front_cloud_cb(self, msg):
        with self._front_lock:
            self._front_cloud = msg

    def _joint_state_cb(self, msg):
        for name, pos in zip(msg.name, msg.position):
            self._joint_pos[name] = pos

    def grasp_is_holding(self):
        """True if a box is currently pinched between the jaws. A finger joint
        held open by the box reads well above an empty (closed-on-air) grasp;
        use the wider-open of the two fingers so an off-centre box (which stops
        only one finger) still counts as held."""
        if not self._joint_pos:
            return False
        gap = max(self._joint_pos.get(j, 0.0) for j in GRIPPER_JOINTS)
        return gap > FINGER_HELD_MIN


    # --- primitives ----------------------------------------------------------
    def move_pose(self, x, y, z, yaw=0.0, cartesian=False, label='',
                  quat_xyzw=None):
        self.get_logger().info(
            f'[arm] -> ({x:.2f},{y:.2f},{z:.2f}) yaw={yaw:.2f} '
            f'{"cartesian " if cartesian else ""}{label}')
        if quat_xyzw is None:
            quat_xyzw = zdown_quat(yaw)
        if cartesian:
            self.arm.move_to_pose(position=(x, y, z), quat_xyzw=quat_xyzw,
                                  cartesian=True, cartesian_fraction_threshold=0.0)
            ok = self.arm.wait_until_executed()
        else:
            ok = self._move_pose_direct(x, y, z, quat_xyzw, label)
        if not ok:
            self.get_logger().warn(f'[arm] motion failed: {label}')
        time.sleep(0.5)
        return ok

    def _move_pose_direct(self, x, y, z, quat_xyzw, label):
        """Non-cartesian pose move that takes the SIMPLE, direct path: solve IK
        seeded from the CURRENT joint state (so the nearest configuration is
        chosen -- no elbow/wrist flip), then plan a short joint-space move to
        it. A bare pose goal lets MoveIt pick any IK solution, which is often a
        far one that swings the joints all the way around. Falls back to a plain
        pose plan if IK or the joint move fails."""
        sol = self.arm.compute_ik(position=(x, y, z), quat_xyzw=quat_xyzw)
        cfg = self._extract_arm_config(sol) if sol is not None else None
        if cfg is not None:
            cfg = self._normalize_roll_config(cfg)
            self.arm.move_to_configuration(cfg)
            if self.arm.wait_until_executed():
                return True
            self.get_logger().warn(
                f'[arm] direct joint move failed for {label}; trying pose plan')
        else:
            self.get_logger().warn(
                f'[arm] IK seed failed for {label}; using pose plan')
        self.arm.move_to_pose(position=(x, y, z), quat_xyzw=quat_xyzw,
                              cartesian=False, cartesian_fraction_threshold=0.0)
        return self.arm.wait_until_executed()

    @staticmethod
    def _extract_arm_config(joint_state):
        """Pull the ARM_JOINTS positions (in order) out of an IK JointState."""
        try:
            return [joint_state.position[joint_state.name.index(j)]
                    for j in ARM_JOINTS]
        except (ValueError, IndexError):
            return None

    def _current_arm_config(self):
        if not all(j in self._joint_pos for j in ARM_JOINTS):
            return None
        return [self._joint_pos[j] for j in ARM_JOINTS]

    def _normalize_roll_config(self, config):
        """For each ROLL joint, replace the target angle with the equivalent
        (+/- 2*pi) that is CLOSEST to the current angle while staying inside the
        joint's +/-2*pi limit. This removes the useless ~2*pi unwind when the
        goal is near one pi and the arm is near the other (same orientation)."""
        cur = self._current_arm_config()
        if cur is None:
            return config
        config = list(config)
        for i in ROLL_JOINT_IDX:
            best = config[i]
            for alt in (config[i] - 2.0 * math.pi, config[i] + 2.0 * math.pi):
                if -ROLL_LIMIT <= alt <= ROLL_LIMIT and abs(alt - cur[i]) < abs(best - cur[i]):
                    best = alt
            config[i] = best
        return config

    def move_config(self, config, label=''):
        """Move to an explicit joint configuration (a direct joint-space plan).
        Roll-joint targets are normalized to the nearest equivalent so the arm
        never unwinds ~2*pi to reach the same orientation."""
        config = self._normalize_roll_config(config)
        self.get_logger().info(f'[arm] -> configuration {label}')
        self.arm.move_to_configuration(config)
        ok = self.arm.wait_until_executed()
        if not ok:
            self.get_logger().warn(f'[arm] motion failed: config {label}')
        time.sleep(0.5)
        return ok

    def gripper(self, pos, label=''):
        self.get_logger().info(f'[gripper] -> {pos} {label}')
        m = JointTrajectory()
        m.joint_names = GRIPPER_JOINTS
        pt = JointTrajectoryPoint()
        pt.positions = [float(pos), float(pos)]
        pt.time_from_start = Duration(sec=1)
        m.points = [pt]
        for _ in range(3):
            self.gripper_pub.publish(m)
            time.sleep(0.4)
        time.sleep(1.0)

    def add_box(self, xy, z_center=None):
        if z_center is None:
            z_center = -0.05 + BOX_SIZE / 2.0     # ground box in base_link frame
        self.arm.add_collision_box(
            id=BOX_ID, size=(BOX_SIZE, BOX_SIZE, BOX_SIZE),
            position=(xy[0], xy[1], z_center), quat_xyzw=(0.0, 0.0, 0.0, 1.0),
            frame_id='base_link')
        time.sleep(0.5)

    # --- perception ------------------------------------------------------------
    def detect_box_pose(self, timeout_sec=5.0, debug_save=False, color='blue'):
        """Wrist (eye-in-hand) detection: move must already be at the scan pose.
        Returns the box centroid (x, y, z) in base_link, or None."""
        return self._detect('wrist', timeout_sec, debug_save, color)

    def detect_box_front(self, timeout_sec=2.0, debug_save=False, color='blue'):
        """Base-mounted front camera detection (used while driving). Returns the
        `color` box centroid (x, y, z) in base_link, or None."""
        return self._detect('front', timeout_sec, debug_save, color)

    def _detect(self, source, timeout_sec, debug_save=False, color='blue'):
        """Waits for a fresh point cloud from the given RGB-D source, HSV-segments
        the blue box, and returns its centroid (x, y, z) in base_link, or None.
        `source` is 'wrist' (/camera/points, camera_link) or 'front'
        (/front_camera/points, front_camera_link)."""
        log = self.get_logger()
        if source == 'front':
            lock, cloud_frame = self._front_lock, 'front_camera_link'
        else:
            lock, cloud_frame = self._cloud_lock, 'camera_link'
        with lock:
            if source == 'front':
                self._front_cloud = None
            else:
                self._latest_cloud = None

        deadline = time.time() + timeout_sec
        cloud = None
        while time.time() < deadline:
            # actively pump this node's own callbacks while waiting (same
            # idiom pymoveit2's wait_until_executed uses) instead of relying
            # solely on the background executor thread to service us.
            rclpy.spin_once(self, timeout_sec=0.2)
            with lock:
                cloud = self._front_cloud if source == 'front' else self._latest_cloud
            if cloud is not None:
                break
        if cloud is None:
            log.error('[detect] no point cloud received before timeout')
            return None

        h, w = cloud.height, cloud.width
        if h <= 1:
            log.error('[detect] point cloud is not organized (height <= 1)')
            return None

        pts = np.array(list(point_cloud2.read_points(
            cloud, field_names=('x', 'y', 'z', 'rgb'), skip_nans=False)))
        x = pts['x'].reshape(h, w)
        y = pts['y'].reshape(h, w)
        z = pts['z'].reshape(h, w)
        rgb_u32 = pts['rgb'].copy().view(np.uint32)
        r = ((rgb_u32 >> 16) & 0xFF).reshape(h, w).astype(np.uint8)
        g = ((rgb_u32 >> 8) & 0xFF).reshape(h, w).astype(np.uint8)
        b = (rgb_u32 & 0xFF).reshape(h, w).astype(np.uint8)
        rgb_img = np.dstack([r, g, b])

        hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
        mask = None
        for lo, hi in COLOR_HSV.get(color, COLOR_HSV['blue']):
            m = cv2.inRange(hsv, lo, hi)
            mask = m if mask is None else cv2.bitwise_or(mask, m)

        if debug_save:
            cv2.imwrite('/tmp/box_rgb_debug.png', cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
            cv2.imwrite('/tmp/box_mask_debug.png', mask)

        valid = (mask.astype(bool) & np.isfinite(x) & np.isfinite(y) & np.isfinite(z))
        n_valid = int(valid.sum())
        if n_valid < MIN_VALID_PIXELS:
            log.error(f'[detect] only {n_valid} valid blue pixels found (need '
                      f'>= {MIN_VALID_PIXELS}) -- box not found')
            return None

        cx, cy, cz = float(x[valid].mean()), float(y[valid].mean()), float(z[valid].mean())
        log.info(f'[detect] {n_valid} px -> centroid ({cx:.3f},{cy:.3f},{cz:.3f}) '
                  f'in {cloud.header.frame_id}')

        # NOTE: the gz-sensors RGBD point cloud generator emits xyz data in
        # the classical (non-optical) camera axis convention -- X-forward,
        # Y-left, Z-up -- even though the message's frame_id names the
        # *optical* frame. Verified empirically: interpreting the points as
        # the (non-optical) camera link frame is what lines up with the known
        # box position after transforming into base_link.
        point = PointStamped()
        point.header = cloud.header
        point.header.frame_id = cloud_frame
        point.point.x, point.point.y, point.point.z = cx, cy, cz
        try:
            tf = self.tf_buffer.lookup_transform(
                'base_link', cloud_frame, rclpy.time.Time(),
                timeout=RclDuration(seconds=1.0))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            log.error(f'[detect] TF lookup base_link <- {cloud_frame} failed: {e}')
            return None
        point_base = tf2_geometry_msgs.do_transform_point(point, tf)
        bx, by, bz = point_base.point.x, point_base.point.y, point_base.point.z

        if abs(bz - EXPECTED_BOX_Z) > 0.02:
            log.warn(f'[detect] detected z={bz:.3f} differs from expected ground '
                      f'box z={EXPECTED_BOX_Z:.3f} by more than 2cm')

        log.info(f'[detect] box in base_link: ({bx:.3f}, {by:.3f}, {bz:.3f})')
        return (bx, by, bz)

    # --- sequence ------------------------------------------------------------
    def _attempt_grasp(self, bx, by):
        """One pre-grasp -> descend -> close on the box at (bx, by). Returns True
        only if a box is actually pinched between the jaws afterwards (verified
        via the finger positions), so a miss is caught instead of assumed."""
        log = self.get_logger()

        # pre-grasp above the box. If planning here fails the box is beyond a
        # comfortable z-down reach; report it so the caller retries/repositions
        # rather than blindly descending from the scan pose and missing.
        if not self.move_pose(bx, by, APPROACH_Z, 0.0, label='pre-grasp'):
            log.warn('[grasp] pre-grasp unreachable -- box too far for a clean grasp')
            return False

        # descend onto the box (remove it from the scene so the jaws may
        # surround it without a false collision), then close.
        self.arm.remove_collision_object(BOX_ID)
        time.sleep(0.3)
        self.move_pose(bx, by, GRASP_Z, 0.0, cartesian=True, label='descend')
        self.gripper(GRIP_CLOSED, 'grasp')

        if not self.grasp_is_holding():
            log.warn('[grasp] jaws closed on air (no box between fingers)')
            return False
        log.info('[grasp] box held between the jaws')
        return True

    def pick_up_box(self):
        """Wrist-cam scan + grasp + lift the box, then hold it in the compact
        CARRY pose. Returns True only after VERIFYING the box is actually held
        (retries the scan+grasp up to MAX_GRASP_ATTEMPTS times, and returns
        False if it never catches the box, so the caller never proceeds as if
        it picked when it didn't). On success the box stays attached to the
        gripper so a caller can drive the base before place_box_down()."""
        log = self.get_logger()
        log.info('=== PICK UP: START ===')
        sx, sy, sz = self.scan_position

        for attempt in range(1, MAX_GRASP_ATTEMPTS + 1):
            log.info(f'--- grasp attempt {attempt}/{MAX_GRASP_ATTEMPTS} ---')

            # 0) open, move the wrist camera over the workspace, and detect the
            #    box afresh (so a box nudged by a previous miss is re-located).
            self.gripper(GRIP_OPEN, 'open')
            self.move_pose(sx, sy, sz, label='scan',
                           quat_xyzw=scan_quat(self.scan_pitch))
            time.sleep(0.5)
            detection = self.detect_box_pose()
            if detection is None:
                log.warn(f'[pick] no box detected at scan pose (attempt {attempt})')
                continue
            bx, by, _bz = detection
            box_xy = (bx, by)

            # box known to MoveIt for pre-grasp/transport awareness + RViz
            self.add_box(box_xy)

            # 1+2) pre-grasp -> descend -> close, and verify the jaws hold it
            if not self._attempt_grasp(bx, by):
                self.arm.remove_collision_object(BOX_ID)
                continue

            # 3) attach the box so MoveIt carries it (and RViz shows it grasped)
            self.add_box(box_xy, z_center=-0.05 + BOX_SIZE / 2.0)
            self.arm.attach_collision_object(
                id=BOX_ID, link_name=GRASP_LINK, touch_links=FINGER_LINKS)
            time.sleep(0.5)

            # 4) lift straight up (cartesian), then tuck into the compact carry
            #    pose. The carry hop is a larger reposition, so use a slow joint
            #    plan rather than cartesian; slow keeps the friction-held box
            #    from being jerked loose.
            self.move_pose(bx, by, APPROACH_Z, 0.0, cartesian=True, label='lift')
            cx, cy, cz = CARRY_POSITION
            self.move_pose(cx, cy, cz, 0.0, cartesian=False, label='carry')

            # 5) confirm the box survived the lift + carry (didn't slip out).
            if not self.grasp_is_holding():
                log.warn('[pick] box slipped during lift/carry -- retrying')
                self.arm.detach_collision_object(BOX_ID)
                self.arm.remove_collision_object(BOX_ID)
                continue

            log.info('=== PICK UP: DONE (box held) ===')
            return True

        log.error(f'=== PICK UP: FAILED after {MAX_GRASP_ATTEMPTS} attempts '
                  f'(no box grasped) ===')
        self.arm.remove_collision_object(BOX_ID)
        return False

    def grab_below(self, grasp_z=GRASP_Z, color='blue', x_offset=FRONT_X_OFFSET):
        """Claw grab: the `color` box has been driven directly UNDER the
        gripper-down ready pose. Take a fresh front-camera read of it, descend
        straight onto that spot (to grasp_z -- raise it for a box on a table),
        close, verify, then lift and tuck into the carry pose. No
        scan/reorientation -- the gripper stays pointing down the whole time.
        Returns True only if the box is actually held (verified via the fingers);
        a miss is retried by the caller re-centring and calling again."""
        log = self.get_logger()
        log.info('=== CLAW GRAB: descend straight down ===')
        self.gripper(GRIP_OPEN, 'open')
        # Fresh read of the box now under the gripper, then descend onto it.
        # x_offset corrects the front camera's forward bias; the default is the
        # value measured for a box on the GROUND. That bias does not hold for a
        # box raised on a table, and the box is only 4.5 cm wide, so applying it
        # there puts the jaws on the box's far edge and shoves it away instead of
        # grasping -- table picks pass a smaller offset.
        det = self.detect_box_front(timeout_sec=1.5, color=color)
        if det is None:
            log.warn('[claw] box not seen for grab')
            return False
        bx = min(GRIPPER_X + 0.03, det[0] + x_offset)
        by = det[1]
        self.move_pose(bx, by, grasp_z, 0.0, cartesian=True,
                       label='claw descend', quat_xyzw=zdown_quat(0.0))
        self.gripper(GRIP_CLOSED, 'grasp')
        if not self.grasp_is_holding():
            log.warn('[claw] jaws closed on air -- lifting to retry')
            self.move_pose(bx, by, READY_Z, 0.0, cartesian=True,
                           label='claw lift-empty', quat_xyzw=zdown_quat(0.0))
            return False
        log.info('[claw] box held between the jaws')

        # attach so MoveIt carries it + RViz shows it, lift straight up, carry
        self.add_box((bx, by), z_center=grasp_z - 0.0575)
        self.arm.attach_collision_object(
            id=BOX_ID, link_name=GRASP_LINK, touch_links=FINGER_LINKS)
        time.sleep(0.5)
        self.move_pose(bx, by, READY_Z, 0.0, cartesian=True,
                       label='claw lift', quat_xyzw=zdown_quat(0.0))
        cx, cy, cz = CARRY_POSITION
        self.move_pose(cx, cy, cz, 0.0, cartesian=False, label='carry')
        if not self.grasp_is_holding():
            log.warn('[claw] box slipped during lift/carry')
            self.arm.detach_collision_object(BOX_ID)
            self.arm.remove_collision_object(BOX_ID)
            return False
        log.info('=== CLAW GRAB: DONE (box held) ===')
        return True

    def place_box_down(self, place_xy=None):
        """Place the currently-held box down at place_xy (base_link frame,
        default PLACE_XY), release, and return the arm home."""
        log = self.get_logger()
        px, py = place_xy if place_xy is not None else PLACE_XY
        place_yaw = math.atan2(py, px)
        log.info('=== PLACE DOWN: START ===')

        # from carry: reposition over the place location (slow joint plan, as
        # for the carry hop), then lower straight down (cartesian).
        self.move_pose(px, py, APPROACH_Z, place_yaw, cartesian=False, label='to place')
        self.move_pose(px, py, GRASP_Z, place_yaw, cartesian=True, label='place-down')
        self.arm.detach_collision_object(BOX_ID)
        time.sleep(0.3)
        self.gripper(GRIP_OPEN, 'release')

        # retreat and go home
        self.move_pose(px, py, APPROACH_Z, place_yaw, cartesian=True, label='retreat')
        self.arm.remove_collision_object(BOX_ID)
        self.move_config(HOME_CONFIG, 'home')
        log.info('=== PLACE DOWN: DONE ===')

    def run(self):
        """Stationary pick-and-place: pick the box up and place it at PLACE_XY
        relative to the current base pose (unchanged external behavior)."""
        self.get_logger().info('=== PICK AND PLACE: START ===')
        if not self.pick_up_box():
            return
        self.place_box_down()
        self.get_logger().info('=== PICK AND PLACE: DONE ===')


def main():
    rclpy.init()
    node = PickAndPlace()
    ex = rclpy.executors.MultiThreadedExecutor(4)
    ex.add_node(node)
    t = threading.Thread(target=node.run, daemon=True)
    # give MoveIt/action servers a moment, then run the sequence
    time.sleep(3.0)
    t.start()
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
