# Explicit type: "pose" (Cartesian tool pose), "joint" (radians), or "".
# Empty type is allowed ONLY for an unused end_pose.
string type
geometry_msgs/PoseStamped pose
# Current group's joint order, same as PlanJointGoal (7 values for one arm).
float64[] joint_positions
