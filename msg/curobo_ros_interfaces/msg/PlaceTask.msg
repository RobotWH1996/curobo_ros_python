string group_name
# Reference pose. Place, preplace and retreat are tool_pose * the relative poses below.
geometry_msgs/PoseStamped tool_pose
# Mixed ordered carrying route before preplace. Each entry is type=pose or type=joint.
curobo_ros_interfaces/MotionWaypoint[] waypoints
# Extra destination AFTER retreat; empty type means no extra destination.
curobo_ros_interfaces/MotionWaypoint end_pose
# true: the three poses below are transforms in base_link.
# false: they are transforms in the tool/ee frame. T = tool_pose * relative.
bool in_base_frame false
# Poses relative to tool_pose. Zero quaternion means identity.
geometry_msgs/Pose place_pose
geometry_msgs/Pose preplace_in_target
geometry_msgs/Pose retreat_in_target
# Standalone Place box size. Its pose relative to the tool is the inverse of place_pose.
# PickPlace reuses Pick's geometry.
geometry_msgs/Vector3 object_size
# Generic gripper joint coordinate (Lycus: metres, open=0.0).
float64 open_position 0.0
# Planned release interpolation duration, seconds.
float64 gripper_duration 1.0
