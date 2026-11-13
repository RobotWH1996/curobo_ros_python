"""Compatibility helpers and state this package keeps about cuRobo objects.

cuRobo solvers are extended by wrapping their methods and caching extra
solvers. That bookkeeping lives here, keyed by the object, and disappears
with it. Nothing in this package adds attributes to a cuRobo object.
"""

from __future__ import annotations

import weakref
from math import isfinite
from types import SimpleNamespace

import torch
from curobo.types import JointState

_STORE = weakref.WeakKeyDictionary()
_FALLBACK = "_adapter_state"


def attached_state(owner):
    """Per-object record. Falls back to one attribute if the object rejects weak refs."""
    try:
        record = _STORE.get(owner)
    except TypeError:
        record = getattr(owner, _FALLBACK, None)
        if record is None:
            record = SimpleNamespace()
            setattr(owner, _FALLBACK, record)
        return record
    if record is None:
        record = SimpleNamespace()
        _STORE[owner] = record
    return record


def replace_method(owner, name: str, fn) -> None:
    """The one place that replaces a method on a cuRobo object."""
    setattr(owner, name, fn)


def contact_ik_cache(motion) -> dict:
    record = attached_state(motion)
    cache = getattr(record, "contact_ik_cache", None)
    if cache is None:
        cache = {}
        record.contact_ik_cache = cache
    return cache


def contact_refine_ik(motion):
    return getattr(attached_state(motion), "contact_refine_ik", None)


def set_contact_refine_ik(motion, solver) -> None:
    attached_state(motion).contact_refine_ik = solver


def last_ik_result(solver):
    return getattr(attached_state(solver), "last_ik_result", None)


def set_last_ik_result(solver, result) -> None:
    attached_state(solver).last_ik_result = result


def collision_scope(cost):
    return getattr(attached_state(cost), "collision_scope", None)


def set_collision_scope(cost, moving) -> None:
    attached_state(cost).collision_scope = SimpleNamespace(moving=moving)


def vertical_costs(motion):
    costs = getattr(attached_state(motion), "vertical_costs", None)
    if not costs:
        raise RuntimeError("OPT 没有可接入的工具姿态代价")
    return costs


def set_vertical_costs(motion, costs) -> None:
    attached_state(motion).vertical_costs = list(costs)


def host_dynamic_limits(transition):
    return getattr(attached_state(transition), "host_dynamic_limits", None)


def set_host_dynamic_limits(transition, limits) -> None:
    attached_state(transition).host_dynamic_limits = limits


def expand_joint_trajectory(
    trajectory: JointState, held_positions: dict[str, float], joint_names: list[str]
) -> JointState:
    """Export the sampled trajectory in controller order with measured held joints."""
    names = trajectory.joint_names
    if not names or len(names) != trajectory.position.shape[-1] or len(set(names)) != len(names):
        raise ValueError("轨迹关节名与位置数组不匹配")
    if len(set(joint_names)) != len(joint_names) or not set(names).issubset(joint_names):
        raise ValueError("完整轨迹关节名重复或缺少规划关节")
    held = []
    for name in joint_names:
        value = 0.0 if name in names else held_positions.get(name, float("nan"))
        if not isfinite(value):
            raise ValueError(f"关节 {name} 缺少有效的保持位置")
        held.append(value)
    positions = trajectory.position
    columns = torch.tensor([joint_names.index(name) for name in names], device=positions.device)
    shape = (*positions.shape[:-1], len(joint_names))
    expanded = positions.new_tensor(held).expand(shape).clone()
    result = JointState.from_position(expanded.index_copy_(-1, columns, positions), joint_names=joint_names)
    for field in ("velocity", "acceleration", "jerk"):
        source = getattr(trajectory, field)
        values = None if source is None else source.new_zeros(shape).index_copy_(-1, columns, source)
        setattr(result, field, values)
    result.dt = trajectory.dt
    return result


def trajectory_is_collision_free(motion, trajectory: JointState) -> bool:
    """Check every active-joint sample against the current graph-planner scene."""
    positions = trajectory.position
    if motion.graph_planner is None or positions is None or positions.numel() == 0:
        return False
    active = trajectory.reorder(list(motion.joint_names))
    samples = active.position.reshape(-1, len(motion.joint_names)).contiguous()
    if not bool(torch.isfinite(samples).all()):
        return False
    graph = motion.graph_planner
    checker = graph.resources if motion.config.graph_planner_algorithm == "rrt" else graph
    feasible = torch.as_tensor(checker.check_samples_feasibility(samples))
    return feasible.numel() == samples.shape[0] and bool(feasible.all())
