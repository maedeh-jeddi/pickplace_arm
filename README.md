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

## Dependencies

Clone the following packages into `~/arm_ws/src/` before building:

```bash
git clone https://github.com/AndrejOrsula/pymoveit2.git src/pymoveit2
git clone https://github.com/ros-controls/gz_ros2_control.git -b humble src/gz_ros2_control
```

After cloning, apply the Gazebo Harmonic patch:

```bash
cd src/gz_ros2_control/gz_ros2_control
git apply ../../../gz_ros2_control_harmonic.patch
```

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

## Mobile base: camera pick, SLAM, Nav2 & autonomous fetch

The arm is mounted on a 4-wheel skid-steer mobile base (`diff_drive_controller`)
with a front-mounted 2D LIDAR (`/scan`, 270° FOV) and a wrist RGB-D camera. Three
levels of autonomy are available:

| Node / launch | Behavior |
| --- | --- |
| `pick_and_place` | Stationary: wrist camera detects the box, arm picks & places it. No pre-set pick pose. |
| `search_and_pick` | Mobile, vision-only: step-and-scan spin to find the box, visually servo the base to it, then pick. |
| `nav_and_pick` | Mobile, map-based: find the box, send a **Nav2** goal (global path + costmap obstacle avoidance) to approach it, then visually servo the last stretch and pick. |

### Extra dependencies (Nav2 + SLAM)

`nav_and_pick` / `autonomous_pick.launch.py` need Nav2, slam_toolbox and
twist_mux. Install them (requires sudo):

```bash
sudo apt update
sudo apt install -y ros-humble-navigation2 ros-humble-nav2-bringup \
    ros-humble-slam-toolbox ros-humble-twist-mux
```

Verify: `ros2 pkg prefix nav2_bringup slam_toolbox twist_mux` should all resolve.
(`pick_and_place` and `search_and_pick` do **not** need these.)

### Running

```bash
cd ~/arm_ws && export GZ_VERSION=harmonic && source install/setup.bash

# Stationary camera pick (box at base_link (0.35, 0)):
ros2 launch pickplace_arm_bringup pick_and_place.launch.py box_x:=0.35 box_y:=0.0

# Mobile vision-only search & pick (box out in the world):
ros2 launch pickplace_arm_bringup search_and_pick.launch.py box_x:=-0.8 box_y:=0.6

# Full autonomous fetch: SLAM + Nav2 navigation + pick:
ros2 launch pickplace_arm_bringup autonomous_pick.launch.py box_x:=-1.2 box_y:=0.8
```

`autonomous_pick.launch.py` composes Gazebo + MoveIt + slam_toolbox + Nav2 +
twist_mux + the `nav_and_pick` behavior, staged with timers. The base avoids
obstacles via Nav2 costmaps; the arm keeps its MoveIt collision-aware planning.

> The arm was shortened (~0.45 m reach), so the base stops with the box ~0.43 m
> ahead — inside both the camera's detection band and the arm's grasp reach.

## Roadmap

- [x] Robot model (URDF/Xacro) with arm and gripper
- [x] Visualization in RViz and Gazebo
- [x] Controller configuration (`ros2_control`)
- [x] Send trajectory goals to move the arm
- [x] MoveIt2 integration for collision-free motion planning
- [x] Separate gripper controller for grasping
- [x] Full pick-and-place logic with a box in Gazebo
- [x] RGB-D camera box detection (no pre-set pick pose)
- [x] Mobile base (4-wheel skid-steer) + LIDAR
- [x] Vision-only mobile search & pick
- [x] SLAM (slam_toolbox) + Nav2 autonomous navigate-and-pick

## Author

Maedeh Jeddi
