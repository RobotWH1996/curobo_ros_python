"""Filter returned IK candidates without changing cuRobo's CUDA batch sizes."""
from __future__ import annotations

from functools import wraps
from typing import Any, Callable
import torch


def install_ik_filter(solver: Any, check: Callable, report: Callable) -> None:
    """Apply CPU checks after IK, before MotionPlanner can use its seed output."""
    original = solver.solve_pose

    @wraps(original)
    def solve_pose(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        if result is None or result.solution is None:
            return result
        # Clone: solver outputs may reference buffers reused by CUDA graphs.
        success = result.success.clone()
        fcl_valid = torch.ones_like(success)
        solution = result.solution.clone()
        flat_success = success.reshape(-1)
        flat_solution = solution.reshape(-1, solution.shape[-1])
        if flat_solution.shape[0] != flat_success.numel():
            raise ValueError('IK solution/success dimensions do not match')
        rows = flat_solution.detach().cpu().numpy()
        rejected = []
        checked = 0
        for index in flat_success.nonzero().flatten().tolist():
            checked += 1
            collision = check(dict(zip(solver.joint_names, rows[index].tolist())))
            if collision is not None:
                flat_success[index] = False
                fcl_valid.reshape(-1)[index] = False
                rejected.append(collision)
        # A fixed number of seeds is required by CUDA graphs. Replace unused
        # slots with a passed candidate; their success flags remain false.
        # This also covers cuRobo goalset paths that consume all seed slots.
        batch = int(getattr(result, 'batch_size', 1) or 1)
        if flat_success.numel() % batch:
            raise ValueError('IK batch/seed dimensions do not match')
        per_batch = flat_success.numel() // batch
        for b in range(batch):
            mask = flat_success[b * per_batch:(b + 1) * per_batch]
            q = flat_solution[b * per_batch:(b + 1) * per_batch]
            good = mask.nonzero().flatten()
            if good.numel():
                q[~mask] = q[good[0]].clone()
        result.success = success
        result.solution = solution
        if result.feasible is not None and result.feasible.shape == success.shape:
            result.feasible = result.feasible & fcl_valid
        if result.js_solution is not None:
            js = result.js_solution.clone()
            full = js.position.reshape(-1, js.position.shape[-1])
            if full.shape[0] != flat_solution.shape[0]:
                raise ValueError('IK joint-state/solution batch dimensions do not match')
            columns = [js.joint_names.index(name) for name in solver.joint_names]
            full[:, columns] = flat_solution
            js.position = full.reshape(js.position.shape)
            result.js_solution = js
        report(checked, rejected)
        return result

    solver.solve_pose = solve_pose
