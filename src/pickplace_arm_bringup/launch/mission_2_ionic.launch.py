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
    """Mission 2 in the IONIC world: sort 3 coloured boxes off a table onto 3
    matching-coloured columns, then park -- same behaviour as mission_2, moved
    into worlds/ionic.sdf (a large cluttered restaurant interior).

    Differences from mission_2.launch.py:
      * Gazebo runs HEADLESS (server only) -- the Ionic world's GUI carries an
        expensive GlobalIlluminationVct plugin that drops RTF to ~0.28 and
        breaks the LIDAR/localisation; headless keeps RTF ~0.9. (Set via env
        below, read by gazebo.launch.py.)
      * World = ionic.sdf, robot spawned at world (6.5, 1.5).
      * Map = maps/ionic.yaml, generated from the world's collision geometry so
        the map frame EQUALS the Gazebo world frame -> mission coordinates are
        world coordinates. AMCL seeded from amcl_ionic.yaml (initial pose
        6.5, 1.5).
      * Table/boxes/columns spawned at the world-frame layout (a +5.70,+1.5
        translation of the warehouse layout into a verified-clear pocket).
      * mission_2_ionic executable (Mission2Ionic, the same logic with the
        Ionic layout constants).
      * RViz off by default (headless host); use_rviz:=true to enable.
    """
    # gazebo.launch.py reads these from the environment; set them before the
    # include is evaluated. setdefault (not a hard assignment) so `HEADLESS=1
    # ros2 launch ...` from the shell still works for reliability testing --
    # but the default here is GUI-on, since watching it run in Gazebo is the
    # normal way to use this launch file. GUI mode drops RTF to ~0.28 (the
    # world's GlobalIlluminationVct plugin is expensive), which is fine to
    # watch but slower and a bit less reliable than headless.
    os.environ.setdefault('WORLD', 'ionic.sdf')
    os.environ.setdefault('HEADLESS', '0')
    os.environ.setdefault('SPAWN_X', '6.5')
    os.environ.setdefault('SPAWN_Y', '1.5')

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
            bringup_share, 'maps', 'ionic.yaml')}])
    amcl = Node(
        package='nav2_amcl', executable='amcl', name='amcl', output='screen',
        parameters=[os.path.join(bringup_share, 'config', 'amcl_ionic.yaml'), sim])
    localization_lifecycle = Node(
        package='nav2_lifecycle_manager', executable='lifecycle_manager',
        name='lifecycle_manager_localization', output='screen',
        parameters=[sim, {'autostart': True, 'bond_timeout': 0.0,
                          'node_names': ['map_server', 'amcl']}])
    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(bringup_share, 'launch', 'nav2.launch.py')))

    # Ionic-world layout (world frame == map frame). Table at world (8.0, 1.5);
    # columns 3.3 m away at world (4.7, 1.5), same relative geometry as the
    # warehouse. Box z = table top 0.10 + box half 0.0225 = 0.1225.
    table = spawn('table', 'table', 8.0, 1.5, 0.05)
    boxes = [spawn('box_red', 'box_red', 8.0, 1.34, 0.1225),
             spawn('box_green', 'box_green', 8.0, 1.50, 0.1225),
             spawn('box_blue', 'box_blue', 8.0, 1.66, 0.1225)]
    columns = [spawn('apriltag_column_1', 'apriltag_column_1', 4.7, 1.20, 0.0),
               spawn('apriltag_column_2', 'apriltag_column_2', 4.7, 1.50, 0.0),
               spawn('apriltag_column_3', 'apriltag_column_3', 4.7, 1.80, 0.0)]

    mission = Node(
        package='pickplace_arm_bringup', executable='mission_2_ionic',
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
        # Force FastDDS to UDP only (shared-memory transport intermittently
        # fails to open a port; see mission_2.launch.py).
        SetEnvironmentVariable(
            'FASTRTPS_DEFAULT_PROFILES_FILE',
            os.path.join(bringup_share, 'config', 'fastdds_udp_only.xml')),
        DeclareLaunchArgument('use_rviz', default_value='false'),
        gazebo,
        move_group,
        TimerAction(period=6.0, actions=[rviz]),
        # Spawn the table (+ columns) first, then the boxes, so a box never
        # lands before the table's collision is live and falls through.
        TimerAction(period=12.0, actions=[table] + columns),
        TimerAction(period=17.0, actions=boxes),
        # Stagger heavy stages; the Ionic world is heavier than the warehouse,
        # so give bring-up a little more room than mission_2 does.
        TimerAction(period=22.0, actions=[map_server, amcl, localization_lifecycle]),
        TimerAction(period=38.0, actions=[nav2]),
        TimerAction(period=65.0, actions=[mission]),
    ])
