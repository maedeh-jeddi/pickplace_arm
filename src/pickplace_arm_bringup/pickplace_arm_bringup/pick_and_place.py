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
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

import tf2_ros

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
# z raised 0.18 -> 0.30 to clear the LIDAR (sits at x=0.14, z=0.06; its scan
# only excludes a 90deg wedge directly BEHIND it, not in front -- at z=0.18
# the held box, hanging below gripper_base since the fingers extend down when
# zdown-oriented, dipped into the LIDAR's forward scan plane and
# self-detected as an obstacle 0.12-0.14m dead ahead, which Nav2 read as
# "collision ahead" with no recovery able to clear it since the box is still
# there afterward). BUT 0.30 turned out unreachable from the post-lift arm
# config without a 180deg base-yaw flip: compute_ik seeded from the real
# post-'claw lift' joint state finds a smooth, same-branch solution up to
# EXACTLY z=0.26 (j1 stays ~0), and NO_IK_SOLUTION or a flipped branch
# (j1 jumps to +/-pi) for everything above that -- `_move_pose_direct`'s
# strict, seeded-IK carry move (deliberately never falls back to an unseeded
# pose plan while holding the box) correctly refused the flip and failed
# every time, so the mission re-opened the gripper each attempt ("release-
# after-failed-carry") and looked like the box kept falling out mid-carry.
# 0.26 is therefore the ceiling for THIS branch -- verified via a
# /compute_ik sweep (x 0.16-0.26, z 0.26-0.30) that 0.26 is the highest z
# reachable without a flip at any nearby x.
CARRY_POSITION = (0.26, 0.00, 0.26)

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
# NOTE: the front camera's old "reads ~2 cm short" fudge factor (FRONT_X_OFFSET)
# is gone. That bias was never a camera error -- it is pure geometry: the camera
# only sees an object's NEAR FACE, so the visible-pixel centroid sits ~half the
# object's depth in front of its true centre (~0.02 for a 4.5 cm box, which is
# where the 2 cm came from). A single constant cannot be right for objects of
# different depths, or for the same box on the ground (top face visible, pulling
# the centroid back) versus raised on a table (near face only) -- which is why
# the table pick needed its own hand-tuned value and still grasped near the box's
# edge. `_detect(depth=...)` now reconstructs the true centre from the measured
# blob extent instead; see there.

# Expected box centroid height in base_link frame (ground plane, see add_box):
# used only as a sanity check against the detected z, not as the commanded z.
EXPECTED_BOX_Z = -0.05 + BOX_SIZE / 2.0

# --- perception ---------------------------------------------------------------
# Fixed vantage point the arm moves to before scanning for the box. Chosen so
# the eye-in-hand camera's frustum covers the reachable table area on the
# ground plane; verify/tune empirically (see detect_box_pose debug dump).
# Position re-derived via /compute_ik sweep after the arm links were
# shortened 30% (old reach ~0.94m -> new ~0.66m); (0.40, 0.60) is no longer
# reachable.
# Pitch: the wrist camera's mount now has ZERO tilt of its own (see
# camera_joint in the URDF -- rigidly aligned with gripper_base), so this
# pitch is the WHOLE downward look angle rather than a delta on top of a
# mount tilt. Its value reproduces the original (pre-mount-tuning) camera
# direction exactly (mount 28.6deg + pitch 55deg = 83.6deg total, all from
# pitch now), so this pose's tuned detection range is unchanged.
SCAN_POSITION = (0.22, 0.00, 0.40)
SCAN_PITCH = math.radians(83.6)

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


def quat_to_matrix(qx, qy, qz, qw):
    """3x3 rotation matrix (numpy) for the xyzw quaternion -- used to rotate a
    whole point cloud into another frame in one vectorised matmul."""
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
    ])


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
                  quat_xyzw=None, strict=False):
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
            ok = self._move_pose_direct(x, y, z, quat_xyzw, label, strict)
        if not ok:
            self.get_logger().warn(f'[arm] motion failed: {label}')
        time.sleep(0.5)
        return ok

    def _move_pose_direct(self, x, y, z, quat_xyzw, label, strict=False):
        """Non-cartesian pose move that takes the SIMPLE, direct path: solve IK
        seeded from the CURRENT joint state (so the nearest configuration is
        chosen -- no elbow/wrist flip), then plan a short joint-space move to
        it. A bare pose goal lets MoveIt pick any IK solution, which is often a
        far one that swings the joints all the way around -- so with
        `strict=True` (used for every move while the arm is holding a box) we
        never fall back to it: a failed seed/joint-move just fails the whole
        call, rather than risking exactly the kind of big uncontrolled swing
        that could visibly rotate the arm and shake the box loose."""
        sol = self.arm.compute_ik(position=(x, y, z), quat_xyzw=quat_xyzw)
        cfg = self._extract_arm_config(sol) if sol is not None else None
        if cfg is not None:
            cfg = self._normalize_roll_config(cfg)
            self.arm.move_to_configuration(cfg)
            if self.arm.wait_until_executed():
                return True
            self.get_logger().warn(f'[arm] direct joint move failed for {label}'
                                   + ('' if strict else '; trying pose plan'))
        else:
            self.get_logger().warn(f'[arm] IK seed failed for {label}'
                                   + ('' if strict else '; using pose plan'))
        if strict:
            return False
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

    def detect_box_front(self, timeout_sec=2.0, debug_save=False, color='blue',
                         depth=None):
        """Base-mounted front camera detection (used while driving). Returns the
        `color` box centroid (x, y, z) in base_link, or None. Pass `depth` (the
        object's known x/y size) whenever the result is used to AIM the gripper
        -- see _detect for why the raw centroid is not the object's centre."""
        return self._detect('front', timeout_sec, debug_save, color, depth)

    def _detect(self, source, timeout_sec, debug_save=False, color='blue',
                depth=None):
        """Waits for a fresh point cloud from the given RGB-D source, HSV-segments
        the blue box, and returns its centroid (x, y, z) in base_link, or None.
        `source` is 'wrist' (/camera/points, camera_link) or 'front'
        (/front_camera/points, front_camera_link).

        With `depth` given (the object's known size along x, front-camera only)
        the returned x/y are the object's CENTRE rather than the mean of its
        visible pixels -- required whenever the gripper is aimed at the result,
        because the two differ by up to half the object's depth."""
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

        # Keep only the LARGEST connected blob of matching pixels. A stray patch
        # of the same colour elsewhere in frame barely shifts the pixel MEAN, but
        # it wrecks the min/max EXTENT the centre estimate below is built from,
        # so it has to go before any geometry is computed.
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            valid.astype(np.uint8), connectivity=8)
        if n_labels > 2:
            biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            valid = (labels == biggest)
            n_valid = int(valid.sum())
            if n_valid < MIN_VALID_PIXELS:
                log.error(f'[detect] largest {color} blob is only {n_valid} px '
                          f'(need >= {MIN_VALID_PIXELS}) -- box not found')
                return None

        # Transform ALL valid points into base_link FIRST, then do every bit of
        # geometry there. Working in base_link (not the cloud frame) makes the
        # centre estimate below independent of how the camera is mounted or
        # aimed: whatever pitch front_camera_joint carries (currently zero --
        # horizontal), the extent is measured along true base_link axes, so the
        # centre is right and stays right if the mount is ever re-angled.
        #
        # NOTE: the gz-sensors RGBD cloud emits xyz in the classical
        # (non-optical) camera convention -- X-forward, Y-left, Z-up in the
        # camera BODY frame -- even though the message frame_id names the optical
        # frame. That convention is relative to the camera body, so it still
        # holds once the body is pitched; the TF base_link<-cloud_frame carries
        # the pitch, so transforming through it lands the points in base_link
        # correctly (verified empirically for the untilted mount; the body-
        # relative argument is why it survives the added tilt).
        try:
            tf = self.tf_buffer.lookup_transform(
                'base_link', cloud_frame, rclpy.time.Time(),
                timeout=RclDuration(seconds=1.0))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            log.error(f'[detect] TF lookup base_link <- {cloud_frame} failed: {e}')
            return None
        q, t = tf.transform.rotation, tf.transform.translation
        pts = np.stack([x[valid], y[valid], z[valid]], axis=1)          # (N,3) cloud
        pts = pts @ quat_to_matrix(q.x, q.y, q.z, q.w).T \
            + np.array([t.x, t.y, t.z])                                 # -> base_link
        bxs, bys, bzs = pts[:, 0], pts[:, 1], pts[:, 2]

        if depth is None or source != 'front':
            if depth is not None:
                log.warn(f'[detect] depth= is only valid for the front camera '
                         f'(got source={source}) -- falling back to the mean')
            # Legacy/coarse reading: the mean of the visible surface. Fine for
            # "how far away is it" (search, drive-in) -- NOT for aiming the
            # gripper, since it sits ~depth/2 in front of the object's centre.
            bx, by = float(bxs.mean()), float(bys.mean())
        else:
            # Reconstruct the object's true CENTRE from its visible surface, in
            # base_link. The depth camera has ~7 mm range noise, which matters
            # for HOW we read the near face:
            #   x: does the blob show one face or two?
            #     * span >= 0.6*depth: the far face is in view too (object BELOW
            #       the camera, e.g. a box on the ground -- near face + top
            #       face), so the full depth is present and the centre is the
            #       midpoint of the x-extent. p5/p95 (not min/max) reject a few
            #       stray pixels, and the two ends' noise biases cancel in the
            #       average, so the midpoint is unbiased.
            #     * otherwise only the NEAR face shows (object at/above camera
            #       height -- EVERY mission_2 table pick and column place): its
            #       points form a thin slab, and the centre is half a known
            #       depth behind that face. Estimate the face with the MEDIAN,
            #       which is unbiased under the symmetric range noise. Do NOT use
            #       a low percentile here: a low percentile of noisy near-face
            #       points sits ~2*sigma proud of the true face (~14 mm at 7 mm
            #       noise), which pulled the grasp ~1.4 cm toward the box's FRONT
            #       EDGE -- invisible on a wide column, obvious on a 4.5 cm box.
            #   y: midpoint of the y extent. A box/column silhouette is symmetric
            #      about its centre from any view angle, so the extent midpoint
            #      (noise-cancelling, unlike the density-weighted mean) is the
            #      centre; p2/p98 reject stray edge pixels.
            p5 = float(np.percentile(bxs, 5))
            p95 = float(np.percentile(bxs, 95))
            if (p95 - p5) >= 0.6 * depth:
                bx = 0.5 * (p5 + p95)
            else:
                bx = float(np.median(bxs)) + depth / 2.0
            by = 0.5 * (float(np.percentile(bys, 2)) + float(np.percentile(bys, 98)))
        bz = float(bzs.mean())

        if abs(bz - EXPECTED_BOX_Z) > 0.02:
            log.warn(f'[detect] detected z={bz:.3f} differs from expected ground '
                      f'box z={EXPECTED_BOX_Z:.3f} by more than 2cm')

        log.info(f'[detect] {n_valid} px -> box in base_link: '
                 f'({bx:.3f}, {by:.3f}, {bz:.3f})')
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
            # strict: never fall back to an unseeded pose plan while holding
            # the box (see _move_pose_direct) -- it can pick a wildly
            # different joint solution and swing the arm around.
            carry_ok = self.move_pose(cx, cy, cz, 0.0, cartesian=False,
                                      label='carry', strict=True)

            # 5) confirm the box survived the lift + carry (didn't slip out).
            if not carry_ok or not self.grasp_is_holding():
                log.warn('[pick] box slipped during lift/carry -- retrying')
                self.arm.detach_collision_object(BOX_ID)
                self.arm.remove_collision_object(BOX_ID)
                # If the carry move itself failed (rather than the box slipping
                # loose on its own), the jaws are still PHYSICALLY closed on it
                # -- open them so the retry starts from an empty gripper.
                if not carry_ok:
                    self.gripper(GRIP_OPEN, 'release-after-failed-carry')
                continue

            log.info('=== PICK UP: DONE (box held) ===')
            return True

        log.error(f'=== PICK UP: FAILED after {MAX_GRASP_ATTEMPTS} attempts '
                  f'(no box grasped) ===')
        self.arm.remove_collision_object(BOX_ID)
        return False

    def grab_below(self, grasp_z=GRASP_Z, color='blue'):
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
        # depth=BOX_SIZE makes the reading the box's CENTRE rather than the
        # centre of the face pointing at the camera: each finger pad is only
        # 1.5 cm deep in x against a 4.5 cm box, so aiming at the visible face
        # (2.25 cm short) puts the pads on the box's near EDGE -- it gets nudged
        # or tipped instead of pinched in the middle.
        det = self.detect_box_front(timeout_sec=1.5, color=color, depth=BOX_SIZE)
        if det is None:
            log.warn('[claw] box not seen for grab')
            return False
        bx = min(GRIPPER_X + 0.03, det[0])   # cap at the arm's comfortable reach
        # Descend on the fixed gripper CENTRELINE (y = GRIPPER_Y = 0), NOT the
        # measured box y. Reaching a sideways y would force the shoulder-yaw
        # joint j1 to rotate -- exactly the arm rotation we must avoid. The base
        # has already centred the box in y (claw_approach, |by| <= CLAW_Y_TOL),
        # and the jaws close from a 0.10 m gap onto a 0.045 m box that is wider
        # than their 0.04 m closed gap, so they physically re-centre it as they
        # shut. So keeping y at 0 both leaves the box centred in the jaws AND
        # keeps the whole descent in the arm's vertical plane (no yaw, no roll).
        by = GRIPPER_Y
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
        # strict: never fall back to an unseeded pose plan while holding the
        # box -- that fallback picks ANY IK solution, including ones that
        # swing the joints all the way around, which can shake the box loose.
        if not self.move_pose(cx, cy, cz, 0.0, cartesian=False, label='carry',
                              strict=True):
            log.warn('[claw] carry move failed -- releasing so the retry '
                     'starts from a clean, empty-gripper state')
            self.arm.detach_collision_object(BOX_ID)
            self.arm.remove_collision_object(BOX_ID)
            # The jaws are still PHYSICALLY closed on the box here (detaching
            # only clears MoveIt's bookkeeping) -- open them too, otherwise the
            # retry re-approaches with a box already clamped in the gripper,
            # which is what turned one failed carry into a fully failed pick.
            self.gripper(GRIP_OPEN, 'release-after-failed-carry')
            return False
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
