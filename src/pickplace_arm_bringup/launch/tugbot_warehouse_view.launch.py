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
    """Spawn the robot + table/boxes/columns into the downloaded Tugbot
    warehouse world (openrobotics "Tugbot in Warehouse" Fuel world) and bring
    up MoveIt so the arm can actually pick/place -- no Nav2/AMCL: everything
    is spawned close enough (inside the yellow hazard-taped square around the
    robot's spawn) that the existing front-camera visual-servo pick/place
    (claw_pick / place_on_column, see mission_2.py) works with the base
    turning in place instead of navigating."""
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

    # table (top at 0.10) 1.8 m BEHIND the robot (spawns at world origin,
    # facing +x, so "behind" is -x); 3 colour boxes on it; 3 matching columns
    # in front. The world has a yellow/black hazard-taped ~4.5x4.5 m square
    # centred exactly on the robot's spawn point (0,0) -- both the table and
    # the columns are kept inside that boundary (+-2.27 m) rather than the
    # original mission_2 spacing, which put the table right on/past its edge.
    table = spawn('table', 'table', -1.80, 0.0, 0.05)
    boxes = [spawn('box_red', 'box_red', -1.80, -0.16, 0.1225),
             spawn('box_green', 'box_green', -1.80, 0.0, 0.1225),
             spawn('box_blue', 'box_blue', -1.80, 0.16, 0.1225)]
    columns = [spawn('apriltag_column_1', 'apriltag_column_1', 1.30, -0.30, 0.0, 0.0),
               spawn('apriltag_column_2', 'apriltag_column_2', 1.30, 0.00, 0.0, 0.0),
               spawn('apriltag_column_3', 'apriltag_column_3', 1.30, 0.30, 0.0, 0.0)]

    demo = Node(
        package='pickplace_arm_bringup', executable='tugbot_demo',
        output='screen', parameters=[sim],
        condition=IfCondition(LaunchConfiguration('run_demo')))

    return LaunchDescription([
        SetEnvironmentVariable('WORLD', 'tugbot_warehouse.sdf'),
        DeclareLaunchArgument('run_demo', default_value='false'),
        gazebo,
        move_group,
        TimerAction(period=10.0, actions=[table] + columns),
        TimerAction(period=15.0, actions=boxes),
        TimerAction(period=20.0, actions=[demo]),
    ])
