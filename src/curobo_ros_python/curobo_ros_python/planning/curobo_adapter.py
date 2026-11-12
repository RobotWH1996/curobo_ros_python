"""State this package keeps about cuRobo objects.

cuRobo solvers are extended by wrapping their methods and caching extra
solvers. That bookkeeping lives here, keyed by the object, and disappears
with it. Nothing in this package adds attributes to a cuRobo object.
"""

from __future__ import annotations

import weakref
from types import SimpleNamespace

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
