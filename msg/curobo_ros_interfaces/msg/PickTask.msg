# Single-arm OPT pick. All distances in metres, quaternions ROS xyzw.
string group_name
# Reference pose. Perception writes the detected pose here.
# Grasp, prepick and lift are tool_pose * the relative poses below.
geometry_msgs/PoseStamped tool_pose
# Mixed ordered route before prepick. Each entry is type=pose or type=joint.
curobo_ros_interfaces/MotionWaypoint[] waypoints
# Extra destination AFTER lift; empty type means no extra destination.
curobo_ros_interfaces/MotionWaypoint end_pose
# true: the three poses below are transforms in base_link.
# false: they are transforms in the tool/ee frame. T = tool_pose * relative.
bool in_base_frame false
# Poses relative to tool_pose. Zero quaternion means identity.
geometry_msgs/Pose pick_pose
geometry_msgs/Pose prepick_in_target
geometry_msgs/Pose lift_in_target
# Object bounding box. Its pose relative to the tool is the inverse of pick_pose.
geometry_msgs/Vector3 object_size
# Generic gripper joint coordinate, in the URDF joint unit (Lycus: metres).
# Lycus open=0.0, fully closed=-0.042; each value is checked against the URDF limit.
float64 open_position 0.0
float64 close_position -0.042
# Planned duration of each open/close interpolation segment, seconds.
float64 gripper_duration 1.0
