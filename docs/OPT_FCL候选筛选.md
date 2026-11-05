# OPT 目标关节候选 FCL 筛选

位姿目标的流程为：IK 返回多组关节解 → FCL 逐组检查 → 通过的解进入原有轨迹优化。筛选统一挂在每个规划组 IK 求解器的 `solve_pose` 返回处，覆盖单目标、多路点、重试和独立 IK。它不更换轨迹优化器内部的球体碰撞代价，也不改变 MPC。

`mesh_collision.py` 负责从 URDF collision 几何构建 FCL BVH、机器人 FK 和环境 mesh；`ik_collision_filter.py` 负责筛选结果和适配固定大小 CUDA 批次；`algorithms/trajopt.py` 负责同步当前关节、反馈和场景。

- 每个候选检查配置中的全机器人自碰撞对和机器人与环境的三角形相交。沿用 `self_collision_ignore`，但恢复左右腕 link5/link7 检查。无 collision 几何的配置 link 会在启动时明确输出。
- 固定手臂采用本次请求的当前状态；夹爪、mimic 和其他固定关节采用最新反馈或配置固定值。无关且未参与碰撞的轮子分支不需要额外反馈。
- 被剔除的候选标记为失败。为了维持 cuRobo 的固定 CUDA seed 数量，空余槽位用同批已通过的候选补齐，不再送入原来碰撞的关节角。全部候选失败时，不调用轨迹优化；现有重试机制可以再次求 IK。
- 显式关节目标在优化前同样检查。轨迹优化可能调整位姿规划的最终构型，因此输出前会再检查实际终点。
- FCL 机器人 BVH 和对象缓存复用。点云更新时重建环境 BVH，并与对应规划组场景同步；静态场景使用 obstacle pose。点云 ESDF 模式从框内源点云额外构建 mesh；只有 ESDF、没有源几何的 `update_world_from_voxel` 明确拒绝，避免漏掉环境检查。
- `last_fcl_candidates` 和规划日志记录最近一轮检查/剔除数量及首个碰撞对。此字段是诊断信息，不是 ROS 服务的新字段。

在 cuRobo 使用的 Python 环境安装依赖：`python -m pip install -r requirements-fcl.txt`。缺少 FCL、几何解析失败或不支持的环境会报错，不会静默放行。首次加载完整 Lycus 网格 BVH 在本机实测约 15 秒，后续复用。

这里是离散目标状态的三角形表面相交筛选，不认证整条路径、连续碰撞或封闭网格完全包含关系。自碰撞忽略表仍需符合实际机械结构。
