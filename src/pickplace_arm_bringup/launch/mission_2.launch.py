import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                            SetEnvironmentVariable, TimerAction)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    """Mission 2 bringup: sort 3 coloured boxes off a table onto 3 matching-
    coloured columns (8/12/10 cm), then park.

    Gazebo (warehouse) + MoveIt + AMCL + Nav2 + the table/boxes/columns
    spawned at fixed spots + the mission_2 behaviour, staged with timers.
    RViz on by default (use_rviz:=false to save RAM). No apriltag_ros: column
    alignment is done with the front camera's colour-blob detector (same as
    the box pick), not an AprilTag read, so the arm/wrist never reorients.
    """
    desc_share = get_package_share_directory('pickplace_arm_description')
    bringup_share = get_package_share_directory('pickplace_arm_bringup')
    models = os.path.join(desc_share, 'models')
    sim = {'use_sim_time': True}

    def spawn(model, name, x, y, z, yaw=0.0):
        return Node(package='ros_gz_sim', executable='create', output='screen',
                    arguments=['-file', os.path.join(models, model, 'model.sdf'),
                               '-name', name, '-x', str(x), '-y', str(y),
                               '-z', str(z), '-Y', str(yaw)])

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(desc_share, 'launch', 'gazebo.launch.py')))

    moveit_config = MoveItConfigsBuilder(
        'pickplace_arm', package_name='pickplace_arm_moveit_config').to_moveit_configs()
    move_group = Node(
        package='moveit_ros_move_group', executable='move_group', output='screen',
        parameters=[moveit_config.to_dict(), sim,
                    {'trajectory_execution.allowed_start_tolerance': 0.1}])

    map_server = Node(
        package='nav2_map_server', executable='map_server', name='map_server',
        output='screen', parameters=[sim, {'yaml_filename': os.path.join(
            bringup_share, 'maps', 'warehouse.yaml')}])
    amcl = Node(
        package='nav2_amcl', executable='amcl', name='amcl', output='screen',
        parameters=[os.path.join(bringup_share, 'config', 'amcl.yaml'), sim])
    localization_lifecycle = Node(
        package='nav2_lifecycle_manager', executable='lifecycle_manager',
        name='lifecycle_manager_localization', output='screen',
        parameters=[sim, {'autostart': True, 'bond_timeout': 0.0,
                          'node_names': ['map_server', 'amcl']}])
    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(bringup_share, 'launch', 'nav2.launch.py')))

    # table (top at 0.10), 3 colour boxes on it, 3 matching-coloured columns
    table = spawn('table', 'table', 2.30, 0.0, 0.05)
    boxes = [spawn('box_red', 'box_red', 2.30, -0.16, 0.1225),
             spawn('box_green', 'box_green', 2.30, 0.0, 0.1225),
             spawn('box_blue', 'box_blue', 2.30, 0.16, 0.1225)]
    # yaw is irrelevant now (square column body, colour-detected, no tag to
    # face) -- kept from an earlier layout. 1 m closer to the table
    # (x=-1.0) than before, no routing around them.
    columns = [spawn('apriltag_column_1', 'apriltag_column_1', -1.0, -0.30, 0.0, 3.14159),
               spawn('apriltag_column_2', 'apriltag_column_2', -1.0, 0.0, 0.0, 3.14159),
               spawn('apriltag_column_3', 'apriltag_column_3', -1.0, 0.30, 0.0, 3.14159)]

    mission = Node(
        package='pickplace_arm_bringup', executable='mission_2', output='screen',
        parameters=[sim])

    rviz = Node(
        package='rviz2', executable='rviz2', name='rviz2', output='screen',
        condition=IfCondition(LaunchConfiguration('use_rviz')),
        prefix=('env -u GTK_PATH -u GTK_EXE_PREFIX -u LOCPATH '
                '-u GDK_PIXBUF_MODULE_FILE -u GDK_PIXBUF_MODULEDIR '
                '-u GIO_MODULE_DIR -u GTK_IM_MODULE_FILE'),
        arguments=['-d', os.path.join(bringup_share, 'config', 'mission.rviz')],
        parameters=[moveit_config.robot_description,
                    moveit_config.robot_description_semantic,
                    moveit_config.robot_description_kinematics,
                    moveit_config.planning_pipelines,
                    moveit_config.joint_limits, sim])

    return LaunchDescription([
        # Force FastDDS to UDP only. Its shared-memory transport intermittently
        # fails to open a port here; when that hits the robot-spawn node it never
        # receives /robot_description, the robot is never spawned, and the whole
        # stack (controllers -> odom -> EKF -> map->base_link) never comes up.
        SetEnvironmentVariable(
            'FASTRTPS_DEFAULT_PROFILES_FILE',
            os.path.join(bringup_share, 'config', 'fastdds_udp_only.xml')),
        DeclareLaunchArgument('use_rviz', default_value='true'),
        gazebo,
        move_group,
        TimerAction(period=6.0, actions=[rviz]),
        # Spawn the table (and columns) FIRST, then the boxes a few seconds
        # later. Creating them in one batch is a race: a box that lands before
        # the table's collision is live falls straight through to the floor
        # (this is what kept "knocking" the green box off before any pick).
        TimerAction(period=10.0, actions=[table] + columns),
        TimerAction(period=15.0, actions=boxes),
        # Stagger the heavy stages well apart -- on this box everything starting
        # at once starves the controller_manager / nav2 lifecycle and the bringup
        # aborts. Let controllers + localization settle before nav2, and nav2
        # before the mission.
        TimerAction(period=20.0, actions=[map_server, amcl, localization_lifecycle]),
        TimerAction(period=35.0, actions=[nav2]),
        TimerAction(period=60.0, actions=[mission]),
    ])
