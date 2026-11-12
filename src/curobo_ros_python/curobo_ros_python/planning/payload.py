"""Box payload geometry shared by GPU planning, FCL, and cloud filtering."""
from __future__ import annotations

from dataclasses import dataclass
import itertools
import numpy as np
from curobo_ros_python.planning.geom import quat_wxyz_to_R


def pose_matrix(xyz, wxyz):
    xyz=np.asarray(xyz,dtype=np.float64).reshape(3)
    q=np.asarray(wxyz,dtype=np.float64).reshape(4)
    if not np.isfinite(xyz).all() or not np.isfinite(q).all() or np.linalg.norm(q)<1e-8:
        raise ValueError('位姿含无效坐标或零四元数')
    q=q/np.linalg.norm(q)
    matrix=np.eye(4)
    matrix[:3,:3]=quat_wxyz_to_R(q)
    matrix[:3,3]=xyz
    return matrix


def outside_box(points, transform, size, padding=0.):
    pts=np.asarray(points)
    local=(pts-transform[:3,3])@transform[:3,:3]
    return ~np.all(np.abs(local)<=np.asarray(size)*.5+padding,axis=1)


def resolve_object_geometry(size, in_tool, use_default_pose, default_size, default_pose):
    """All-zero requested size selects config; pose selection is explicit."""
    size=np.asarray(size,dtype=np.float64).reshape(3)
    if np.all(size==0):
        size=np.asarray(default_size,dtype=np.float64).reshape(3)
    if not np.isfinite(size).all() or np.any(size<=0):
        raise ValueError('工件尺寸须为正；省略 object_size 时请先配置 pick_place_default_object_size')
    if use_default_pose:
        if not np.allclose(in_tool,np.eye(4),atol=1e-8,rtol=0):
            raise ValueError('已填写 object_in_tool 时须设置 use_default_object_in_tool=false，避免忽略请求几何')
        values=np.asarray(default_pose,dtype=np.float64).reshape(7)
        in_tool=pose_matrix(values[:3],[values[6],values[3],values[4],values[5]])
    return size,in_tool


@dataclass
class Payload:
    group: str
    size: np.ndarray
    in_tool: np.ndarray
    origin: np.ndarray

    def __post_init__(self):
        self.size=np.asarray(self.size,dtype=np.float64).reshape(3)
        if not np.isfinite(self.size).all() or np.any(self.size<=0):
            raise ValueError('工件必须有有限正尺寸 object_size')

    @property
    def tool(self):
        from curobo_ros_python.planning.robot_profile import active_profile

        return active_profile().ee_link_name(self.group)

    @property
    def link(self):
        from curobo_ros_python.planning.robot_profile import active_profile

        return active_profile().payload_link_name(self.group)

    def spheres(self):
        # 27 circumscribed cells cover the whole box, not just its surface.
        cell=self.size/3
        centers=np.array(list(itertools.product((-1.,0.,1.),repeat=3)))*cell
        centers=centers@self.in_tool[:3,:3].T+self.in_tool[:3,3]
        return np.c_[centers,np.full(27,np.linalg.norm(cell)*.5+.001)].astype(np.float32)


def configure_payload_links(robot):
    """Reserve inactive sphere slots on both tools before CUDA graph capture."""
    from curobo_ros_python.planning.robot_profile import active_profile

    profile = active_profile()
    kin=robot['robot_cfg']['kinematics']
    # A robot without modeled gripper jaws (K100's hands) keeps its own config.
    bases = {profile.gripper_base_name(side) for side in ('left', 'right')}
    if not profile.jaw_links or not bases.issubset(kin.get('collision_link_names',[])):
        return robot
    links=kin.setdefault('extra_links',{})
    spheres=kin.setdefault('extra_collision_spheres',{})
    ignored=kin.setdefault('self_collision_ignore',{})
    for group in ('left_arm','right_arm'):
        tool = profile.ee_link_name(group)
        if tool not in kin.get('tool_frames',[]):
            continue
        name = profile.payload_link_name(group)
        links[name]=dict(link_name=name,parent_link_name=tool,joint_name=name+'_joint',
                         joint_type='FIXED',fixed_transform=[0,0,0,1,0,0,0])
        spheres[name]=27
        if name not in kin['collision_link_names']: kin['collision_link_names'].append(name)
        side=group.split('_')[0]
        contacts=[tool, profile.gripper_base_name(side), *profile.jaw_link_names(side)]
        # Payload touches fingers/tool by design, not proximal arm links.
        ignored[name]=list(dict.fromkeys(contacts))
        for link in contacts:
            ignored.setdefault(link,[])
            if name not in ignored[link]: ignored[link].append(name)
    return robot
