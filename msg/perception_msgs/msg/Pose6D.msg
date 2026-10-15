# ------------------------------------------------------------------------------
# 文件名称：Pose6D.msg
# 创建日期：2026-09-03
# 创建人：Charles
# 功能：目标 6DoF 位姿（机器人基座系）。同一帧时间戳在 Pose6DArray.header。
# 话题：/obj_poses
#
# 修改记录：
#   2026-09-03  Charles  创建文件
#   2026-09-08  Charles  补充文件信息头，注释统一为中文
#   2026-09-30  Charles  增加 track_id，与点云字段对齐；去掉 class_name，只留 class_id
# ------------------------------------------------------------------------------

int32 valid          # 1 有效，-1 无效
float32 x            # 位置 x（米）
float32 y            # 位置 y（米）
float32 z            # 位置 z（米）
float32 qx           # 四元数 x
float32 qy           # 四元数 y
float32 qz           # 四元数 z
float32 qw           # 四元数 w
int32 label          # ObjectLabel：5 BOX，11 ROD，12 HOLE，0 Background
float32 score        # 检测置信度
int32 track_id       # 本帧目标号。>0 与点云 track_id 对应；0 无对应。dynamic 跨帧稳定，static 仅本帧有效