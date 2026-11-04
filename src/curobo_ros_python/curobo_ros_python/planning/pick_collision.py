"""Scoped contact permissions and payload collision state on existing OPT models."""
from __future__ import annotations

from contextlib import contextmanager
import numpy as np


def contact_links(group):
    side=group.split('_')[0]
    return (side+'_gripper_left_jaw',side+'_gripper_right_jaw')


def models(planner):
    from curobo_ros_python.planning.algorithms.trajopt import _robot_models
    result=[]
    seen=set()
    for motion in planner._planners.values():
        for model in _robot_models(motion):
            kin=model.config.kinematics_config
            if id(kin.link_spheres) not in seen:
                seen.add(id(kin.link_spheres));result.append(kin)
    return result


def set_payload(planner,payload):
    """Update all solver copies in place; preserve CUDA graph tensor addresses."""
    import torch
    bindings=[]
    for kin in models(planner):
        for group in ('left_arm','right_arm'):
            name='pick_place_'+group+'_payload'
            if name not in kin.link_name_to_idx_map: continue
            indices=kin.get_sphere_index_from_link_name(name)
            if len(indices)!=27: raise RuntimeError(f'{name} 未预留27个碰撞球')
            data=np.zeros((27,4),dtype=np.float32);data[:,3]=-100.
            if payload is not None and payload.group==group: data=payload.spheres()
            bindings.append((kin,indices,torch.as_tensor(data,device=kin.link_spheres.device,dtype=kin.link_spheres.dtype)))
    if payload is not None and not bindings: raise RuntimeError('模型没有携物碰撞球预留，请重新启动节点')
    for kin,indices,data in bindings: kin.link_spheres[:,indices,:]=data
    planner._mesh_checker.payload=payload
    for motion in planner._planners.values():
        if motion.graph_planner is not None: motion.graph_planner.reset_buffer()


@contextmanager
def contact_scope(planner,group,enabled=False,disable_payload=False,suspend_fcl=False):
    """Only finger WORLD contacts relaxed in FCL; self contacts remain checked."""
    checker=planner._mesh_checker
    previous=getattr(checker,'environment_ignore_links',set())
    old_payload=getattr(checker,'payload',None)
    old_skip=getattr(checker,'skip_fcl_checks',False)
    saved=[]
    try:
        checker.skip_fcl_checks=suspend_fcl
        if disable_payload: checker.payload=None
        if enabled:
            links=list(contact_links(group))
            if disable_payload: links.append('pick_place_'+group+'_payload')
            for kin in models(planner):
                for link in links:
                    if link not in kin.link_name_to_idx_map: raise ValueError(f'模型缺少接触连杆 {link}')
                    indices=kin.get_sphere_index_from_link_name(link)
                    radii=kin.link_spheres[:,indices,3].clone()
                    saved.append((kin,indices,radii))
                    kin.link_spheres[:,indices,3]=-100.
            checker.environment_ignore_links=set(previous)|set(contact_links(group))
        yield
    finally:
        for kin,indices,radii in reversed(saved): kin.link_spheres[:,indices,3]=radii
        checker.environment_ignore_links=previous
        checker.payload=old_payload
        checker.skip_fcl_checks=old_skip
