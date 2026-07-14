import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg_description = get_package_share_directory('pickplace_arm_description')

    xacro_file = os.path.join(pkg_description, 'urdf', 'pickplace_arm.urdf.xacro')
    # Default to the warehouse world (bigger, varied obstacles); override with
    # the WORLD env var pointing at another .sdf under worlds/ if needed.
    world_name = os.environ.get('WORLD', 'warehouse.sdf')
    world_file = os.path.join(pkg_description, 'worlds', world_name)

    robot_description = {
        'robot_description': ParameterValue(
            Command(['xacro ', xacro_file, ' use_gazebo:=true']), value_type=str
        )
    }

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('ros_gz_sim'),
                'launch',
                'gz_sim.launch.py'
            )
        ),
        # gz_version 8 = Gazebo Harmonic
        launch_arguments={
            'gz_args': '-r ' + world_file,
            'gz_version': '8',
        }.items(),
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[robot_description, {'use_sim_time': True}],
    )

    spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-topic', '/robot_description',
            '-name', 'pickplace_arm',
            # base_link (the root) sits one wheel-radius above the ground so
            # the wheels touch the floor; base_footprint hangs below it.
            '-z', '0.05'
        ],
        output='screen',
    )

    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster', '--controller-manager', '/controller_manager'],
        output='screen',
    )

    arm_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['arm_controller', '--controller-manager', '/controller_manager'],
        output='screen',
    )

    gripper_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['gripper_controller', '--controller-manager', '/controller_manager'],
        output='screen',
    )

    diff_drive_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['diff_drive_controller', '--controller-manager', '/controller_manager'],
        output='screen',
    )

    delayed_joint_state_broadcaster = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=spawn_entity,
            on_exit=[joint_state_broadcaster_spawner],
        )
    )

    delayed_arm_controller = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[arm_controller_spawner],
        )
    )

    delayed_gripper_controller = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=arm_controller_spawner,
            on_exit=[gripper_controller_spawner],
        )
    )

    delayed_diff_drive_controller = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=gripper_controller_spawner,
            on_exit=[diff_drive_controller_spawner],
        )
    )

    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            # Simulation clock -> ROS, so use_sim_time nodes get a time source
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            '/camera/image@sensor_msgs/msg/Image[gz.msgs.Image',
            '/camera/depth_image@sensor_msgs/msg/Image[gz.msgs.Image',
            '/camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
            '/camera/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked',
            '/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
            '/imu@sensor_msgs/msg/Imu[gz.msgs.IMU',
            # Front base-mounted RGB-D camera (box detection while driving)
            '/front_camera/image@sensor_msgs/msg/Image[gz.msgs.Image',
            '/front_camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
            '/front_camera/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked',
        ],
        output='screen',
    )

    # robot_localization EKF: fuses wheel odometry (forward velocity) with the
    # IMU (heading) to publish a stable odom -> base_link transform. The
    # diff_drive controller's own odom TF is disabled (enable_odom_tf: false)
    # so this is the single source of that transform.
    ekf = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[
            os.path.join(pkg_description, 'config', 'ekf.yaml'),
            {'use_sim_time': True},
        ],
    )

    return LaunchDescription([
        gazebo,
        robot_state_publisher,
        spawn_entity,
        delayed_joint_state_broadcaster,
        delayed_arm_controller,
        delayed_gripper_controller,
        delayed_diff_drive_controller,
        bridge,
        ekf,
    ])
