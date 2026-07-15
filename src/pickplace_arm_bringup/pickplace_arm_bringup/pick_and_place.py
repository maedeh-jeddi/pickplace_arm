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
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from geometry_msgs.msg import PointStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

import tf2_ros
import tf2_geometry_msgs  # noqa: F401  (registers PointStamped transform support)

from pymoveit2 import MoveIt2

ARM_JOINTS = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
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

# Compact "carry" pose: box held low and centered over the base so it rides
# stably while the mobile base drives to the delivery point.
CARRY_POSITION = (0.26, 0.00, 0.18)

# Neutral / "home" arm configuration (joint angles j1..j6): a raised,
# forward-curling "cobra" posture (user-tuned). The arm also SPAWNS in this
# pose -- see the initial_value params in pickplace_arm.gazebo.xacro.
HOME_CONFIG = [0.0, 0.4, 0.75, 3.14, -1.4, 0.0]

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

# HSV bounds for the box's blue material (ambient/diffuse 0.1 0.5 0.9 ->
# sRGB ~(26,128,230) -> HSV ~(104,227,230)), widened for shading/noise.
HSV_LOWER = (95, 120, 60)
HSV_UPPER = (115, 255, 255)
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
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.get_logger().info('Pick-and-place node ready')

    def _cloud_cb(self, msg):
        with self._cloud_lock:
            self._latest_cloud = msg

    def _front_cloud_cb(self, msg):
        with self._front_lock:
            self._front_cloud = msg

    # --- primitives ----------------------------------------------------------
    def move_pose(self, x, y, z, yaw=0.0, cartesian=False, label='',
                  quat_xyzw=None):
        self.get_logger().info(
            f'[arm] -> ({x:.2f},{y:.2f},{z:.2f}) yaw={yaw:.2f} '
            f'{"cartesian " if cartesian else ""}{label}')
        if quat_xyzw is None:
            quat_xyzw = zdown_quat(yaw)
        self.arm.move_to_pose(position=(x, y, z), quat_xyzw=quat_xyzw,
                              cartesian=cartesian,
                              cartesian_fraction_threshold=0.0)
        ok = self.arm.wait_until_executed()
        if not ok:
            self.get_logger().warn(f'[arm] motion failed: {label}')
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
    def detect_box_pose(self, timeout_sec=5.0, debug_save=False):
        """Wrist (eye-in-hand) detection: move must already be at the scan pose.
        Returns the box centroid (x, y, z) in base_link, or None."""
        return self._detect('wrist', timeout_sec, debug_save)

    def detect_box_front(self, timeout_sec=2.0, debug_save=False):
        """Base-mounted front camera detection (used while driving). Returns the
        box centroid (x, y, z) in base_link, or None."""
        return self._detect('front', timeout_sec, debug_save)

    def _detect(self, source, timeout_sec, debug_save=False):
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
        mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)

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
    def pick_up_box(self):
        """Wrist-cam scan + grasp + lift the box, then hold it in the compact
        CARRY pose. Returns True on success. Leaves the box attached to the
        gripper so a caller can drive the base before place_box_down()."""
        log = self.get_logger()
        log.info('=== PICK UP: START ===')

        # 0) scan pose: move the wrist camera over the workspace and detect
        #    the box's position -- no pre-defined pick pose is used.
        self.gripper(GRIP_OPEN, 'open')
        sx, sy, sz = self.scan_position
        self.move_pose(sx, sy, sz, label='scan',
                       quat_xyzw=scan_quat(self.scan_pitch))
        time.sleep(0.5)
        detection = self.detect_box_pose()
        if detection is None:
            log.error('No box detected at scan pose -- aborting pick.')
            return False
        bx, by, _bz = detection
        box_xy = (bx, by)

        # box known to MoveIt for pre-grasp/transport awareness + RViz display
        self.add_box(box_xy)

        # 1) pre-grasp above the box
        self.move_pose(bx, by, APPROACH_Z, 0.0, label='pre-grasp')

        # 2) descend onto the box (remove it from the scene so the jaws may
        #    surround it without a false collision), then grasp
        self.arm.remove_collision_object(BOX_ID)
        time.sleep(0.3)
        self.move_pose(bx, by, GRASP_Z, 0.0, cartesian=True, label='descend')
        self.gripper(GRIP_CLOSED, 'grasp')

        # 3) attach the box so MoveIt carries it (and RViz shows it grasped)
        self.add_box(box_xy, z_center=-0.05 + BOX_SIZE / 2.0)
        self.arm.attach_collision_object(
            id=BOX_ID, link_name=GRASP_LINK, touch_links=FINGER_LINKS)
        time.sleep(0.5)

        # 4) lift straight up (cartesian), then tuck into the compact carry
        #    pose. The carry hop is a larger reposition, so use a slow joint
        #    plan (max_velocity 0.15) rather than cartesian (which can only
        #    partially complete a long straight-line path); slow keeps the
        #    friction-held box from being jerked loose.
        self.move_pose(bx, by, APPROACH_Z, 0.0, cartesian=True, label='lift')
        cx, cy, cz = CARRY_POSITION
        self.move_pose(cx, cy, cz, 0.0, cartesian=False, label='carry')
        log.info('=== PICK UP: DONE (box held) ===')
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
        self.arm.move_to_configuration(HOME_CONFIG)
        self.arm.wait_until_executed()
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
