import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                            TimerAction)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    """One-shot autonomous warehouse mission bringup.

    Gazebo (warehouse) + MoveIt move_group + AMCL localization (on the saved
    map) + Nav2 + RViz (map, costmaps, scan, Nav2 path, both cameras, MoveIt
    planning) + the target box + the mission behavior (search -> pick ->
    deliver -> park), staged with timers.

    RViz is on by default; disable it (e.g. to save RAM) with use_rviz:=false.
    Override the box start location, e.g.:
      ros2 launch pickplace_arm_bringup mission.launch.py box_x:=-2.0 box_y:=3.0
    """
    desc_share = get_package_share_directory('pickplace_arm_description')
    bringup_share = get_package_share_directory('pickplace_arm_bringup')
    box_sdf = os.path.join(desc_share, 'models', 'target_box', 'model.sdf')

    box_x = LaunchConfiguration('box_x')
    box_y = LaunchConfiguration('box_y')
    sim = {'use_sim_time': True}

    # Gazebo (warehouse + controllers + EKF + sensors)
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(desc_share, 'launch', 'gazebo.launch.py')))

    # MoveIt move_group (no RViz)
    moveit_config = MoveItConfigsBuilder(
        'pickplace_arm', package_name='pickplace_arm_moveit_config'
    ).to_moveit_configs()
    move_group = Node(
        package='moveit_ros_move_group', executable='move_group',
        output='screen', parameters=[moveit_config.to_dict(), sim])

    map_server = Node(
        package='nav2_map_server', executable='map_server', name='map_server',
        output='screen',
        parameters=[sim, {'yaml_filename': os.path.join(
            bringup_share, 'maps', 'warehouse.yaml')}])

    amcl = Node(
        package='nav2_amcl', executable='amcl', name='amcl', output='screen',
        parameters=[os.path.join(bringup_share, 'config', 'amcl.yaml'), sim])

    localization_lifecycle = Node(
        package='nav2_lifecycle_manager', executable='lifecycle_manager',
        name='lifecycle_manager_localization', output='screen',
        parameters=[sim, {'autostart': True,
                          'bond_timeout': 0.0,
                          'node_names': ['map_server', 'amcl']}])

    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(bringup_share, 'launch', 'nav2.launch.py')))

    spawn_box = Node(
        package='ros_gz_sim', executable='create', output='screen',
        arguments=['-file', box_sdf, '-name', 'target_box',
                   '-x', box_x, '-y', box_y, '-z', '0.0225'])

    mission = Node(
        package='pickplace_arm_bringup', executable='mission',
        output='screen', parameters=[sim])

    # RViz: map + costmaps + scan + Nav2 path + both cameras + MoveIt planning.
    # The 'prefix' strips the VS Code *snap* GTK/GIO/LOCPATH env vars, which
    # otherwise make RViz load a glibc-incompatible snap libpthread and crash
    # (symbol lookup error __libc_pthread_init). It gets the MoveIt params so
    # the MotionPlanning display works.
    rviz = Node(
        package='rviz2', executable='rviz2', name='rviz2', output='screen',
        condition=IfCondition(LaunchConfiguration('use_rviz')),
        prefix=('env -u GTK_PATH -u GTK_EXE_PREFIX -u LOCPATH '
                '-u GDK_PIXBUF_MODULE_FILE -u GDK_PIXBUF_MODULEDIR '
                '-u GIO_MODULE_DIR -u GTK_IM_MODULE_FILE'),
        arguments=['-d', os.path.join(bringup_share, 'config', 'mission.rviz')],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
            moveit_config.joint_limits,
            sim,
        ])

    return LaunchDescription([
        DeclareLaunchArgument('box_x', default_value='2.5'),
        DeclareLaunchArgument('box_y', default_value='-1.5'),
        DeclareLaunchArgument('use_rviz', default_value='true'),
        gazebo,
        move_group,
        # RViz a little after Gazebo so /robot_description + move_group exist.
        TimerAction(period=6.0, actions=[rviz]),
        TimerAction(period=9.0, actions=[spawn_box]),
        TimerAction(period=14.0, actions=[map_server, amcl,
                                          localization_lifecycle]),
        TimerAction(period=20.0, actions=[nav2]),
        TimerAction(period=38.0, actions=[mission]),
    ])
