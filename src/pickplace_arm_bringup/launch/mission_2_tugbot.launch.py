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
    """Mission 2 in the downloaded Tugbot-warehouse world (OpenRobotics
    "Tugbot in Warehouse" Fuel world): sort 3 coloured boxes off a table onto
    3 matching-coloured columns, then park -- same behaviour and layout as
    mission_2.launch.py (this world's open aisle is clear at those exact
    coordinates), just pointed at a different world + map + AMCL config, and
    running the colour-detection-gated Mission2Tugbot instead of Mission2.

    Map = maps/tugbot_warehouse.yaml, built by SLAM with the robot starting
    at its spawn pose, so map frame == world frame == the plain warehouse
    layout's coordinates unchanged. AMCL seeded from amcl_tugbot.yaml
    (initial pose 0,0,0, matching spawn).
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
            bringup_share, 'maps', 'tugbot_warehouse.yaml')}])
    amcl = Node(
        package='nav2_amcl', executable='amcl', name='amcl', output='screen',
        parameters=[os.path.join(bringup_share, 'config', 'amcl_tugbot.yaml'), sim])
    localization_lifecycle = Node(
        package='nav2_lifecycle_manager', executable='lifecycle_manager',
        name='lifecycle_manager_localization', output='screen',
        parameters=[sim, {'autostart': True, 'bond_timeout': 0.0,
                          'node_names': ['map_server', 'amcl']}])
    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(bringup_share, 'launch', 'nav2.launch.py')))

    # Same relative layout as the plain warehouse mission -- table 2.30 m
    # ahead of spawn, columns 1.0 m the other way -- this room's open aisle
    # is clear at those coordinates (SLAM map verified).
    #
    # Everything was scaled up for the Husky A200 + FR3. The old props were
    # sized for a 0.38 m-long chassis with a 0.66 m-reach arm whose gripper
    # worked 0.05 m off the floor; the Husky is 0.99 m long and the FR3's base
    # alone sits 0.384 m up, so the old 0.10 m table and 0.08-0.16 m columns
    # were entirely below the front camera's horizon (0.332 m) and could not
    # be seen at all at approach range.
    #   table   0.18x0.60x0.10  ->  0.25x1.00x0.30   (top at z=0.30)
    #   boxes   0.045 cube      ->  0.06 cube        (still well inside the
    #                                                 Franka Hand's 0.08 m jaw)
    #   columns 0.12sq, 8/12/16 ->  0.20sq, 30/40/50 cm
    # Spawn z is the model ORIGIN: the table box is centred, so z = height/2;
    # a box sits on the table top at 0.30 + 0.06/2; columns are modelled from
    # their base so they spawn at z=0.
    table = spawn('table', 'table', 2.30, 0.0, 0.15)
    boxes = [spawn('box_red', 'box_red', 2.30, -0.22, 0.33),
             spawn('box_green', 'box_green', 2.30, 0.0, 0.33),
             spawn('box_blue', 'box_blue', 2.30, 0.22, 0.33)]
    # Column y-spacing widened 0.30 -> 0.45 to match the wider (0.20) columns
    # and the much wider Husky, so the base can square up on one without its
    # bumper overlapping a neighbour.
    columns = [spawn('apriltag_column_1', 'apriltag_column_1', -1.0, -0.45, 0.0, 3.14159),
               spawn('apriltag_column_2', 'apriltag_column_2', -1.0, 0.0, 0.0, 3.14159),
               spawn('apriltag_column_3', 'apriltag_column_3', -1.0, 0.45, 0.0, 3.14159)]

    mission = Node(
        package='pickplace_arm_bringup', executable='mission_2_tugbot',
        output='screen', parameters=[sim])

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
        SetEnvironmentVariable('WORLD', 'tugbot_warehouse.sdf'),
        SetEnvironmentVariable(
            'FASTRTPS_DEFAULT_PROFILES_FILE',
            os.path.join(bringup_share, 'config', 'fastdds_udp_only.xml')),
        DeclareLaunchArgument('use_rviz', default_value='false'),
        gazebo,
        move_group,
        # Startup is paced against the SIM CLOCK settling, not just node
        # readiness. This world is a big downloaded Fuel scene and takes a long
        # time to stream in; while it does, /clock jumps backwards repeatedly
        # (measured: 561 "Detected jump back in time" warnings, from t+5 s
        # right through to t+62 s). AMCL does not tolerate that -- it throws
        # tf2::ExtrapolationException looking up lidar_link->odom and ABORTS
        # (SIGABRT, exit -6), taking localization down and stranding the
        # mission before it can navigate anywhere.
        #
        # So localization must not be activated until after the clock is
        # monotonic. These periods put map_server/amcl comfortably past the
        # measured settle point, with nav2 and the mission staged behind it.
        # Verified afterwards by sampling /clock: strictly increasing.
        TimerAction(period=6.0, actions=[rviz]),
        TimerAction(period=12.0, actions=[table] + columns),
        TimerAction(period=17.0, actions=boxes),
        TimerAction(period=75.0, actions=[map_server, amcl, localization_lifecycle]),
        TimerAction(period=95.0, actions=[nav2]),
        TimerAction(period=120.0, actions=[mission]),
    ])
