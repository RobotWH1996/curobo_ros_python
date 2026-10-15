# 左右臂笛卡尔目标，发到同一个话题。has_left / has_right 为 false 时该臂钉在当前末端。
# Pose 默认四元数是 (0,0,0,1)，不能靠四元数是否为空判断缺失。
std_msgs/Header header
bool has_left
bool has_right
geometry_msgs/PoseStamped left
geometry_msgs/PoseStamped right
