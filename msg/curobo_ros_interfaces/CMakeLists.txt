cmake_minimum_required(VERSION 3.8)
project(curobo_ros_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)

find_package(action_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/JointWaypoint.msg"
  "msg/MpcTarget.msg"
  "msg/MpcArmResult.msg"
  "msg/MpcPlan.msg"
  "srv/PlanJointGoal.srv"
  "srv/PlanPoseGoal.srv"
  "srv/ComputeIK.srv"
  "action/FollowMpc.action"
  DEPENDENCIES geometry_msgs std_msgs trajectory_msgs action_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
