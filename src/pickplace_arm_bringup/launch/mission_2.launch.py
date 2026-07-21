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
    """Mission 2 bringup: sort 3 coloured boxes off a table onto 3 AprilTag
    columns (10/15/20 cm), then park.

    Gazebo (warehouse) + MoveIt + AMCL + Nav2 + apriltag_ros (front camera) +
    the table/boxes/columns spawned at fixed spots + the mission_2 behaviour,
    staged with timers. RViz on by default (use_rviz:=false to save RAM).
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

    # AprilTag detector on the WRIST/gripper camera (publishes TF
    # camera_optical_link -> tag36h11:<id>). The tags sit flat on the column
    # tops, so the arm dips the wrist camera over each column to read its tag.
    # size = tag edge in metres (the 0.08 m tag box, texture ~0.9 -> ~0.072).
    # detector.decimate=1.0 (no downsampling) for the small close-range tags.
    apriltag = Node(
        package='apriltag_ros', executable='apriltag_node', name='apriltag',
        output='screen',
        parameters=[sim, {'family': '36h11', 'size': 0.072, 'max_hamming': 0,
                          'detector.decimate': 1.0}],
        remappings=[('image_rect', '/camera/image'),
                    ('camera_info', '/camera/camera_info')])

    # table (top at 0.10), 3 colour boxes on it, 3 AprilTag columns
    table = spawn('table', 'table', 2.30, 0.0, 0.05)
    boxes = [spawn('box_red', 'box_red', 2.30, -0.16, 0.1225),
             spawn('box_green', 'box_green', 2.30, 0.0, 0.1225),
             spawn('box_blue', 'box_blue', 2.30, 0.16, 0.1225)]
    # columns rotated 180 deg so their tag face points +x, toward the robot
    # approaching from the open table side -- no routing around them. 1 m closer
    # to the table (x=-1.0) than before.
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
        # before the mission; apriltag (CPU-heavy) starts last, near placement.
        TimerAction(period=20.0, actions=[map_server, amcl, localization_lifecycle]),
        TimerAction(period=35.0, actions=[nav2]),
        TimerAction(period=70.0, actions=[apriltag]),
        TimerAction(period=60.0, actions=[mission]),
    ])
