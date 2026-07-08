# pickplace_arm

A 6-DOF robot arm with a 2-finger gripper for pick-and-place simulation in ROS2 Humble and Gazebo Harmonic.

## Overview

`pickplace_arm` is a ROS2 package that models a 6 degrees-of-freedom robotic arm equipped with a simple 2-finger gripper. The goal of the project is to pick up a box and place it at a target location inside a Gazebo simulation, using the `ros2_control` framework and MoveIt2 for motion planning.

## Environment

- ROS2 Humble
- Gazebo Harmonic (`gz-sim8`)
- `gz_ros2_control` (`GazeboSimSystem` plugin), built from source against Harmonic
- `ros_gz` (Harmonic variant: `ros-humble-ros-gzharmonic-*`)

## Package structure

​```
pickplace_arm_description/
├── urdf/
│   ├── pickplace_arm.urdf.xacro     # Arm + gripper model (8 joints)
│   └── pickplace_arm.gazebo.xacro   # gz_ros2_control system + RGB-D camera (use_gazebo:=true)
├── worlds/
│   └── pickplace.sdf                # Harmonic world with the Sensors system
├── config/
│   └── arm_controllers.yaml         # Controller configuration
└── launch/
    ├── gazebo.launch.py             # Spawn arm in Gazebo Harmonic + start controllers
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

Build the workspace (the `gz_ros2_control` plugin must be compiled against
Harmonic via the `GZ_VERSION` environment variable), then launch:

​```bash
cd ~/arm_ws
export GZ_VERSION=harmonic
colcon build
source install/setup.bash
ros2 launch pickplace_arm_description gazebo.launch.py
​```

In a separate terminal, verify the controllers are active:

​```bash
cd ~/arm_ws
source install/setup.bash
ros2 control list_controllers
​```

Expected output: `joint_state_broadcaster`, `arm_controller` and
`gripper_controller` should all be `active`.

### MoveIt motion planning in Gazebo

To plan and execute collision-free motions from RViz that actually drive the
arm in the Gazebo Harmonic simulation, launch the combined bringup (Gazebo +
`move_group` + RViz):

​```bash
cd ~/arm_ws
export GZ_VERSION=harmonic
source install/setup.bash
ros2 launch pickplace_arm_moveit_config gazebo_moveit.launch.py
​```

Then in RViz use the **MotionPlanning** panel: drag the goal state, *Plan*,
then *Execute* — the arm moves in Gazebo. `demo.launch.py` still exists for
standalone MoveIt with mock hardware (no Gazebo).

> Note: RViz must be started from a normal (non-snap) terminal. If it exits
> with `undefined symbol: __libc_pthread_init`, a snap runtime has injected
> `/snap/...` into `LD_LIBRARY_PATH`; clear those entries and relaunch.

## Roadmap

- [x] Robot model (URDF/Xacro) with arm and gripper
- [x] Visualization in RViz and Gazebo
- [x] Controller configuration (`ros2_control`)
- [x] Send trajectory goals to move the arm
- [x] MoveIt2 integration for collision-free motion planning
- [x] Separate gripper controller for grasping
- [ ] Full pick-and-place logic with a box in Gazebo

## Author

Maedeh Jeddi
