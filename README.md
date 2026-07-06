# pickplace_arm

A 6-DOF robot arm with a 2-finger gripper for pick-and-place simulation in ROS2 Humble and Gazebo Classic.

## Overview

`pickplace_arm` is a ROS2 package that models a 6 degrees-of-freedom robotic arm equipped with a simple 2-finger gripper. The goal of the project is to pick up a box and place it at a target location inside a Gazebo simulation, using the `ros2_control` framework and (later) MoveIt2 for motion planning.

## Environment

- ROS2 Humble
- Gazebo Classic
- `gazebo_ros2_control` (GazeboSystem plugin)

## Package structure

​```
pickplace_arm_description/
├── urdf/
│   └── pickplace_arm.urdf.xacro     # Arm + gripper model (8 joints)
├── config/
│   └── arm_controllers.yaml         # Controller configuration
└── launch/
    ├── gazebo.launch.py             # Spawn arm in Gazebo + start controllers
    └── display.launch.py            # Visualize the model in RViz
​```

## Robot description

- 6 revolute joints (`joint1` ... `joint6`) for the arm
- 2 prismatic joints (`left_finger_joint`, `right_finger_joint`) for the gripper
- All joints use a `position` command interface with `position` and `velocity` state interfaces

## Controllers

- `joint_state_broadcaster` — publishes joint states
- `arm_controller` — a `joint_trajectory_controller` driving all 8 joints

## How to run

Build the package and launch the simulation:

​```bash
cd ~/arm_ws
colcon build --packages-select pickplace_arm_description
source install/setup.bash
ros2 launch pickplace_arm_description gazebo.launch.py
​```

In a separate terminal, verify the controllers are active:

​```bash
cd ~/arm_ws
source install/setup.bash
ros2 control list_controllers
​```

Expected output: both `joint_state_broadcaster` and `arm_controller` should be `active`.

## Roadmap

- [x] Robot model (URDF/Xacro) with arm and gripper
- [x] Visualization in RViz and Gazebo
- [x] Controller configuration (`ros2_control`)
- [x] Send trajectory goals to move the arm
- [ ] MoveIt2 integration for collision-free motion planning
- [ ] Separate gripper controller for grasping
- [ ] Full pick-and-place logic with a box in Gazebo

## Author

Maedeh Jeddi
