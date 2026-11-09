"""Restrict collision queries to pairs involving this plan's moving links."""
from collections import defaultdict
from dataclasses import replace
from functools import wraps
from typing import Iterable
import xml.etree.ElementTree as ET


def moving_links(joints: Iterable, active_joints: Iterable[str], extra_links=()) -> frozenset[str]:
    """Include every descendant affected by an active joint, including tools."""
    joints = list(joints)
    active = set(active_joints)
    unknown = active - {joint.get('name') for joint in joints}
    if unknown:
        raise ValueError(f'Unknown motion joints: {sorted(unknown)}')
    children = defaultdict(list)
    roots = []
    for joint in joints:
        child = joint.find('child').get('link')
        children[joint.find('parent').get('link')].append(child)
        if joint.get('name') in active:
            roots.append(child)
    for link in extra_links:
        children[link['parent_link_name']].append(link['link_name'])
    result = set()
    while roots:
        link = roots.pop()
        if link not in result:
            result.add(link)
            roots.extend(children[link])
    return frozenset(result)


def configure_motion_collisions(robot: dict, active_joints: Iterable[str]) -> frozenset[str]:
    """Keep moving/static self pairs; omit only static/static pairs at startup."""
    from curobo.config_io import join_path
    from curobo.content import get_assets_path
    kin = robot['robot_cfg']['kinematics']
    urdf = join_path(str(get_assets_path()), kin['urdf_path'])
    moving = moving_links(ET.parse(urdf).getroot().findall('joint'), active_joints,
                          kin.get('extra_links', {}).values())
    if not moving:
        raise ValueError('Motion collision policy requires active robot joints')
    static = set(kin.get('collision_link_names', ())) - moving
    ignored = kin.setdefault('self_collision_ignore', {})
    for link in sorted(static):
        ignored[link] = sorted(set(ignored.get(link, ())) | (static - {link}))
    return moving


def install_motion_collision(owner, moving: frozenset[str]) -> None:
    """Mask only WORLD queries before CUDA capture; self-collision spheres stay intact."""
    import torch
    pending = [owner]
    seen_owners = set()
    rollouts = []
    while pending:
        item = pending.pop()
        if item is None or id(item) in seen_owners:
            continue
        seen_owners.add(id(item))
        core = getattr(item, 'core', None)
        if core is not None:
            rollouts.extend(core.get_all_rollout_instances())
            rollouts.extend(getattr(core, 'additional_metrics_rollouts', {}).values())
        elif hasattr(item, 'get_all_rollout_instances'):
            rollouts.extend(item.get_all_rollout_instances())
        pending.extend(getattr(item, name, None) for name in
                       ('ik_solver', 'trajopt_solver', 'seed_ik_solver', 'graph_planner'))
    seen = set()
    costs = []
    for rollout in rollouts:
        model = rollout.transition_model.robot_model.config.kinematics_config
        flags = torch.zeros(model.num_links, device=model.link_sphere_idx_map.device, dtype=torch.bool)
        for link in moving:
            if link in model.link_name_to_idx_map:
                flags[model.link_name_to_idx_map[link]] = True
        for manager in rollout._cost_manager_list:
            cost = manager.get_cost('scene_collision')
            if cost is None or id(cost) in seen:
                continue
            seen.add(id(cost))
            if getattr(cost, '_moving_collision_links', None) is not None:
                if cost._moving_collision_links != moving:
                    raise RuntimeError('GPU collision cost is shared by different motion groups')
                continue
            original = cost.forward

            def scoped_forward(state, *args, _forward=original, _flags=flags, _model=model, **kwargs):
                # Derive membership from the live sphere map, including edited spheres.
                geometry = state.robot_collision_geometry
                sphere_map = _model.link_sphere_idx_map if geometry is None else geometry.link_sphere_idx_map
                index = sphere_map.to(dtype=torch.long)
                active = _flags[index]
                spheres = state.robot_spheres
                radius = torch.where(active, spheres[..., 3], -100.)
                masked = replace(state, robot_spheres=torch.cat((spheres[..., :3], radius[..., None]), -1))
                return _forward(masked, *args, **kwargs)

            cost.forward = wraps(original)(scoped_forward)
            cost._moving_collision_links = moving
            costs.append(cost)
    if not costs:
        raise RuntimeError('No GPU scene collision costs available for motion scoping')
    owner._moving_collision_costs = costs
