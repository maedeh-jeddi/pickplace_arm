#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from pymoveit2 import MoveIt2
import time
from shape_msgs.msg import SolidPrimitive
from moveit_msgs.msg import CollisionObject
from std_msgs.msg import Header
from geometry_msgs.msg import Pose


class PickAndPlace(Node):
    def __init__(self):
        super().__init__('pick_and_place')

        self.callback_group = ReentrantCallbackGroup()

        self.moveit2_arm = MoveIt2(
            node=self,
            joint_names=['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'],
            base_link_name='base_link',
            end_effector_name='gripper_base',
            group_name='arm',
            callback_group=self.callback_group,
        )
        self.moveit2_arm.max_velocity = 0.3
        self.moveit2_arm.max_acceleration = 0.3

        self.moveit2_gripper = MoveIt2(
            node=self,
            joint_names=['left_finger_joint', 'right_finger_joint'],
            base_link_name='base_link',
            end_effector_name='gripper_base',
            group_name='gripper',
            callback_group=self.callback_group,
        )

        self.collision_pub = self.create_publisher(
            CollisionObject, '/collision_object', 10)

        self.get_logger().info('Pick and Place node initialized')

    def open_gripper(self):
        self.get_logger().info('Opening gripper...')
        self.moveit2_gripper.move_to_configuration([0.03, 0.03])
        self.moveit2_gripper.wait_until_executed()
        time.sleep(1.0)

    def close_gripper(self):
        self.get_logger().info('Closing gripper...')
        self.moveit2_gripper.move_to_configuration([0.0, 0.0])
        self.moveit2_gripper.wait_until_executed()
        time.sleep(1.0)

    def move_arm_joints(self, joints, sec=3):
        """Move arm using joint space (reliable)"""
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
        from builtin_interfaces.msg import Duration
        if not hasattr(self, 'arm_pub'):
            self.arm_pub = self.create_publisher(
                JointTrajectory, '/arm_controller/joint_trajectory', 10)
        msg = JointTrajectory()
        msg.joint_names = ['joint1','joint2','joint3','joint4','joint5','joint6']
        pt = JointTrajectoryPoint()
        pt.positions = joints
        pt.time_from_start = Duration(sec=sec)
        msg.points = [pt]
        self.arm_pub.publish(msg)
        self.get_logger().info(f'Moving to joints: {[round(j,2) for j in joints]}')
        time.sleep(sec + 1.0)

    def move_gripper_direct(self, positions, sec=2):
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
        from builtin_interfaces.msg import Duration
        if not hasattr(self, 'gripper_pub'):
            self.gripper_pub = self.create_publisher(
                JointTrajectory, '/gripper_controller/joint_trajectory', 10)
        msg = JointTrajectory()
        msg.joint_names = ['left_finger_joint', 'right_finger_joint']
        pt = JointTrajectoryPoint()
        pt.positions = positions
        pt.time_from_start = Duration(sec=sec)
        msg.points = [pt]
        self.gripper_pub.publish(msg)
        self.get_logger().info(f'Gripper -> {positions}')
        time.sleep(sec + 0.5)

    def add_box(self, name, x, y, z, size=0.05):
        self.get_logger().info(f'Adding box at ({x}, {y}, {z})')
        obj = CollisionObject()
        obj.header = Header()
        obj.header.frame_id = 'base_link'
        obj.header.stamp = self.get_clock().now().to_msg()
        obj.id = name
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = [size, size, size]
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z
        pose.orientation.w = 1.0
        obj.primitives = [primitive]
        obj.primitive_poses = [pose]
        obj.operation = CollisionObject.ADD
        for _ in range(5):
            self.collision_pub.publish(obj)
            time.sleep(0.2)

    def run(self):
        self.get_logger().info('=== Pick and Place START ===')

        # Box position - جلوی بازو روی زمین
        BOX_X, BOX_Y, BOX_Z = 0.30, 0.0, 0.10

        # Joint poses - از tf2_echo محاسبه شده
        HOME       = [0.0,  0.0,  0.0,  0.0,  0.0,  0.0]
        ABOVE_BOX  = [0.0,  0.7,  0.4,  0.0, -0.2,  0.0]  # بالای جعبه
        AT_BOX     = [0.0,  0.9,  0.6,  0.0, -0.3,  0.0]  # کنار جعبه
        LIFT       = [0.0,  0.5,  0.3,  0.0, -0.1,  0.0]  # بالا بردن
        PLACE      = [1.2,  0.7,  0.4,  0.0, -0.2,  0.0]  # محل گذاشتن

        # Step 1: Add box
        self.add_box('target_box', BOX_X, BOX_Y, BOX_Z)

        # Step 2: Home
        self.get_logger().info('Step 1: Home...')
        self.move_arm_joints(HOME)

        # Step 3: Open gripper
        self.get_logger().info('Step 2: Open gripper...')
        self.move_gripper_direct([0.03, 0.03])

        # Step 4: Above box
        self.get_logger().info('Step 3: Moving above box...')
        self.move_arm_joints(ABOVE_BOX)

        # Step 5: Down to box
        self.get_logger().info('Step 4: Going down to box...')
        self.move_arm_joints(AT_BOX)

        # Step 6: Close gripper
        self.get_logger().info('Step 5: Grasping...')
        self.move_gripper_direct([0.0, 0.0])

        # Step 7: Lift
        self.get_logger().info('Step 6: Lifting...')
        self.move_arm_joints(LIFT)

        # Step 8: Place
        self.get_logger().info('Step 7: Moving to place...')
        self.move_arm_joints(PLACE)

        # Step 9: Release
        self.get_logger().info('Step 8: Releasing...')
        self.move_gripper_direct([0.03, 0.03])

        # Step 10: Home
        self.get_logger().info('Step 9: Going home...')
        self.move_arm_joints(HOME)

        self.get_logger().info('=== Pick and Place COMPLETE ===')


def main():
    rclpy.init()
    node = PickAndPlace()
    executor = rclpy.executors.MultiThreadedExecutor(2)
    executor.add_node(node)
    time.sleep(2.0)

    import threading
    thread = threading.Thread(target=node.run)
    thread.start()

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()
        thread.join()


if __name__ == '__main__':
    main()
