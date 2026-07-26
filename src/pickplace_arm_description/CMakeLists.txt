cmake_minimum_required(VERSION 3.8)
project(pickplace_arm_description)

find_package(ament_cmake REQUIRED)

# No `meshes` directory: the Husky A200 and FR3/Franka Hand geometry is
# referenced straight out of clearpath_platform_description and
# franka_description via $(find ...) / package:// rather than copied here.
install(
  DIRECTORY urdf launch rviz config worlds models
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
