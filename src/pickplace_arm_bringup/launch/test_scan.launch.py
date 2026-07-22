import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    """Minimal rig to tune the wrist-camera column-top scan: gazebo + MoveIt +
    apriltag(on the wrist /camera) + one column spawned right in front of the
    stationary robot. No nav/AMCL/mission -- iterate the arm scan pose fast."""
    desc_share = get_package_share_directory('pickplace_arm_description')
    sim = {'use_sim_time': True}

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(desc_share, 'launch', 'gazebo.launch.py')))

    moveit_config = MoveItConfigsBuilder(
        'pickplace_arm', package_name='pickplace_arm_moveit_config').to_moveit_configs()
    move_group = Node(
        package='moveit_ros_move_group', executable='move_group', output='screen',
        parameters=[moveit_config.to_dict(), sim,
                    {'trajectory_execution.allowed_start_tolerance': 0.1}])

    apriltag = Node(
        package='apriltag_ros', executable='apriltag_node', name='apriltag',
        output='screen',
        parameters=[sim, {'family': '36h11', 'size': 0.072, 'max_hamming': 0,
                          'detector.decimate': 1.0}],
        remappings=[('image_rect', '/camera/image'),
                    ('camera_info', '/camera/camera_info')])

    # No column spawned here -- the sweep driver spawns/removes one column at
    # a time at the real approach spot (0.45, 0, 0) via ros_gz_sim create / gz
    # service remove, so the arm can be tested at its ACTUAL fixed scan pose
    # (mission_2 never re-aims the arm's y -- the base does that via Nav2).
    return LaunchDescription([
        gazebo,
        move_group,
        TimerAction(period=12.0, actions=[apriltag]),
    ])
