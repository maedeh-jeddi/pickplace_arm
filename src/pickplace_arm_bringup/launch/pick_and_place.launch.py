import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    desc_share = get_package_share_directory('pickplace_arm_description')
    moveit_share = get_package_share_directory('pickplace_arm_moveit_config')
    box_sdf = os.path.join(desc_share, 'models', 'target_box', 'model.sdf')

    # Box spawn position (base_link frame, ground level) -- override to prove
    # the pick-and-place node is finding the box via the camera, not a
    # hardcoded pose, e.g.:
    #   ros2 launch pickplace_arm_bringup pick_and_place.launch.py box_x:=0.35 box_y:=-0.25
    box_x_arg = DeclareLaunchArgument('box_x', default_value='0.60')
    box_y_arg = DeclareLaunchArgument('box_y', default_value='0.00')
    box_x = LaunchConfiguration('box_x')
    box_y = LaunchConfiguration('box_y')

    # Gazebo Harmonic + ros2_control + move_group + RViz
    gazebo_moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(moveit_share, 'launch', 'gazebo_moveit.launch.py')
        )
    )

    # Spawn the physics-enabled box on the ground once Gazebo is up.
    # base_link sits 0.05 m above the ground, so world z = 0.0225 places the
    # 0.045 m cube on the floor at base_link (box_x, box_y, -0.0275).
    spawn_box = TimerAction(
        period=8.0,
        actions=[Node(
            package='ros_gz_sim', executable='create', output='screen',
            arguments=['-file', box_sdf, '-name', 'target_box',
                       '-x', box_x, '-y', box_y, '-z', '0.0225'],
        )],
    )

    # Run the pick-and-place sequence once everything is up.
    pick_and_place = TimerAction(
        period=16.0,
        actions=[Node(
            package='pickplace_arm_bringup', executable='pick_and_place',
            output='screen', parameters=[{'use_sim_time': True}],
        )],
    )

    return LaunchDescription(
        [box_x_arg, box_y_arg, gazebo_moveit, spawn_box, pick_and_place])
