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

    # Box spawn position (base_link frame, ground level) -- deliberately far
    # from the robot's start pose by default, so the run actually exercises
    # the search/drive behavior instead of finding the box without moving.
    # Override to try other distances/directions, e.g.:
    #   ros2 launch pickplace_arm_bringup search_and_pick.launch.py box_x:=2.0 box_y:=-1.0
    box_x_arg = DeclareLaunchArgument('box_x', default_value='1.3')
    box_y_arg = DeclareLaunchArgument('box_y', default_value='0.4')
    box_x = LaunchConfiguration('box_x')
    box_y = LaunchConfiguration('box_y')

    # Gazebo Harmonic + ros2_control (arm + diff-drive base) + move_group + RViz
    gazebo_moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(moveit_share, 'launch', 'gazebo_moveit.launch.py')
        )
    )

    # Spawn the physics-enabled box on the ground once Gazebo is up.
    spawn_box = TimerAction(
        period=8.0,
        actions=[Node(
            package='ros_gz_sim', executable='create', output='screen',
            arguments=['-file', box_sdf, '-name', 'target_box',
                       '-x', box_x, '-y', box_y, '-z', '0.0225'],
        )],
    )

    # Run the autonomous search-and-pick sequence once everything is up.
    search_and_pick = TimerAction(
        period=16.0,
        actions=[Node(
            package='pickplace_arm_bringup', executable='search_and_pick',
            output='screen', parameters=[{'use_sim_time': True}],
        )],
    )

    return LaunchDescription(
        [box_x_arg, box_y_arg, gazebo_moveit, spawn_box, search_and_pick])
