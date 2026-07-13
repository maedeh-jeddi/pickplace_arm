import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                            TimerAction)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """One-shot autonomous navigate-and-pick bringup.

    Composes, staged with timers so each layer starts after its inputs exist:
      Gazebo + ros2_control + MoveIt (+ RViz)  -> robot, /scan, arm planning
      slam_toolbox                             -> map, map->odom TF
      Nav2 + twist_mux                         -> path planning + obstacle avoid
      target box spawn                         -> the object to fetch
      nav_and_pick node                        -> the autonomous behavior

    Override the box location, e.g.:
      ros2 launch pickplace_arm_bringup autonomous_pick.launch.py \
          box_x:=-1.2 box_y:=0.8
    """
    desc_share = get_package_share_directory('pickplace_arm_description')
    moveit_share = get_package_share_directory('pickplace_arm_moveit_config')
    bringup_share = get_package_share_directory('pickplace_arm_bringup')
    box_sdf = os.path.join(desc_share, 'models', 'target_box', 'model.sdf')

    box_x = LaunchConfiguration('box_x')
    box_y = LaunchConfiguration('box_y')

    gazebo_moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(moveit_share, 'launch', 'gazebo_moveit.launch.py')))

    slam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(bringup_share, 'launch', 'slam.launch.py')))

    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(bringup_share, 'launch', 'nav2.launch.py')))

    spawn_box = Node(
        package='ros_gz_sim', executable='create', output='screen',
        arguments=['-file', box_sdf, '-name', 'target_box',
                   '-x', box_x, '-y', box_y, '-z', '0.0225'])

    # A walled enclosure gives slam_toolbox continuous structure to localize
    # against (the LIDAR needs features; open ground yields no map). Spawned
    # around the robot so mapping works from the start.
    wall_sdf = os.path.join(desc_share, 'models', 'wall', 'model.sdf')
    walls = [
        Node(package='ros_gz_sim', executable='create', output='screen',
             arguments=['-file', wall_sdf, '-name', n,
                        '-x', x, '-y', y, '-z', '0.3'] + yaw)
        for (n, x, y, yaw) in [
            ('wall_n', '0', '3.0', []),
            ('wall_s', '0', '-3.0', []),
            ('wall_e', '3.0', '0', ['-Y', '1.5708']),
            ('wall_w', '-3.0', '0', ['-Y', '1.5708']),
        ]
    ]

    nav_and_pick = Node(
        package='pickplace_arm_bringup', executable='nav_and_pick',
        output='screen', parameters=[{'use_sim_time': True}])

    return LaunchDescription([
        # Box default: within the camera's ~1.1 m find-by-spin range so the
        # robot detects it from the start, then Nav2-drives to ~0.6 m in front
        # of it before the visual servo + arm pick take over. (Boxes beyond
        # ~1.1 m need the blind-exploration creep, which is less reliable.)
        DeclareLaunchArgument('box_x', default_value='0.95'),
        DeclareLaunchArgument('box_y', default_value='0.0'),
        gazebo_moveit,
        # Walls + box once Gazebo is up.
        TimerAction(period=9.0, actions=walls + [spawn_box]),
        # SLAM + Nav2 once Gazebo is publishing /scan and TF.
        TimerAction(period=12.0, actions=[slam]),
        TimerAction(period=16.0, actions=[nav2]),
        # Behavior once the full stack is fully up (move_group ready).
        TimerAction(period=34.0, actions=[nav_and_pick]),
    ])
