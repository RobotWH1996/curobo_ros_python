# 一个 MPC 周期的双臂规划结果。left / right 轨迹时间轴相同。
std_msgs/Header header
bool success
string message
float32 solve_time
MpcArmResult left
MpcArmResult right
