"""规划算法共用的位姿小工具和锁关节配置。TrajOpt / MPC 都从这里引，互不依赖。"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from curobo.config_io import join_path, load_yaml
from curobo.content import get_robot_configs_path
from curobo.types import Pose

from curobo_ros_python.planning.geom import quat_angle_wxyz
from curobo_ros_python.planning.groups import GROUP_JOINTS
from curobo_ros_python.planning.gripper import GRIPPER_JOINTS


def squeeze_pose(pose: Pose) -> Pose:
    return Pose(
        position=pose.position.reshape(-1, 3)[:1].contiguous(),
        quaternion=pose.quaternion.reshape(-1, 4)[:1].contiguous(),
        normalize_rotation=False,
    )


def pose_xyz_wxyz(pose: Pose) -> Tuple[List[float], List[float]]:
    p = pose.position.detach().reshape(-1).cpu().numpy()
    q = pose.quaternion.detach().reshape(-1).cpu().numpy()
    return [float(p[0]), float(p[1]), float(p[2])], [
        float(q[0]),
        float(q[1]),
        float(q[2]),
        float(q[3]),
    ]


def pose_error(current: Pose, target: Pose) -> Tuple[float, float]:
    c_xyz, c_wxyz = pose_xyz_wxyz(current)
    t_xyz, t_wxyz = pose_xyz_wxyz(target)
    pos = float(np.linalg.norm(np.array(t_xyz) - np.array(c_xyz)))
    ang = quat_angle_wxyz(c_wxyz, t_wxyz)
    return pos, ang


def robot_cfg_with_locked_joints(
    robot: str,
    current: Optional[Dict[str, float]],
    extra_lock_names: Optional[Sequence[str]] = None,
) -> Tuple[Dict, Dict[str, float]]:
    """锁住不下发的关节：躯干/头，以及本次没用的那一侧手臂。

    TrajOpt / MPC 都会优化 cspace 里所有未锁关节。只规划左臂却让右臂参与时，
    求解器会把「靠近目标」分给右臂，而且自由度多一倍。
    """
    data = load_yaml(join_path(str(get_robot_configs_path()), robot))
    kinematics = data["robot_cfg"]["kinematics"]
    cspace = kinematics["cspace"]
    all_names = list(cspace["joint_names"])
    arm_names = set(GROUP_JOINTS.get("both_arms") or [])
    extra = set(extra_lock_names or [])
    if not arm_names and not extra:
        return data, {}
    defaults = list(cspace.get("default_joint_position") or [])
    locks = dict(kinematics.get("lock_joints") or {})
    for name in GRIPPER_JOINTS:
        if name in all_names:
            locks[name] = float((current or {}).get(name, locks.get(name, 0.0)))
    for i, name in enumerate(all_names):
        if name in locks:
            continue
        if name in arm_names and name not in extra:
            continue
        if current and name in current:
            locks[name] = float(current[name])
        elif i < len(defaults):
            locks[name] = float(defaults[i])
        else:
            locks[name] = 0.0
    kinematics["lock_joints"] = locks
    return data, locks


def pop_cached_mesh(checker, name: str):
    """按名字取出已上传的网格。

    cuRobo 的 Warp 缓存只认名字。点云每拍都叫同一个名字时，
    不先取出，新几何会被丢掉，碰撞一直用第一帧。
    调用方在加载成功后再丢掉返回值，旧 BVH 才会释放。
    加载失败时要把返回值交回 ``restore_cached_mesh``：底层会先清空场景槽，
    只把对象留在本地变量里并不会把碰撞环境装回去。
    """
    meshes = getattr(getattr(checker, "data", None), "meshes", None)
    cache = getattr(meshes, "wp_cache", None)
    if not cache:
        return None
    return cache.pop(name, None)


def restore_cached_mesh(checker, name: str, old) -> None:
    """加载失败后把取出的网格装回缓存和场景槽。

    ``load_collision_model`` 先把环境槽的 enable/count/names 清掉，再上传新网格。
    失败时新网格可能已经进了缓存，场景槽却已经空了。这里先放回旧网格，
    再用它的 Warp id 把槽位写回去，最后才释放上传失败的那份。
    """
    meshes = getattr(getattr(checker, "data", None), "meshes", None)
    cache = getattr(meshes, "wp_cache", None)
    if cache is None:
        return
    failed = cache.pop(name, None)
    if old is None:
        if failed is not None:
            del failed
        return
    cache[name] = old
    pose = Pose.from_list([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], meshes.device_cfg)
    meshes.update_from_warp_id(int(old.mesh_id), name, w_obj_pose=pose)
    idx = meshes.names[0].index(name)
    lower, upper = old.get_bounds()
    meshes.dims[0, idx, :3] = upper - lower
    if failed is not None and failed is not old:
        del failed
