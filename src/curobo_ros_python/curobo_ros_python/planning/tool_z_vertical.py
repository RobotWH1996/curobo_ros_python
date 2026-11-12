"""World-Z tool-axis preference for OPT; endpoint poses and yaw stay independent."""
from contextlib import contextmanager
import math
import torch
from curobo._src.cost.cost_tool_pose import ToolPoseCost


VERTICAL_TOLERANCE_DEG = 15.
ROUTE_RADIUS_M = .10
ROUTE_WEIGHT_RATIO = 3.


def signed_axis_error(quaternion, sign, cos_limit=1.):
    """Smooth inclination cost, invariant to quaternion sign and world yaw."""
    z_world = 1. - 2. * (quaternion[..., 1].square() + quaternion[..., 2].square())
    return torch.clamp(cos_limit - sign * z_world, min=0.)


def route_excursion(position, radius=ROUTE_RADIUS_M):
    """Squared excess beyond a soft start/end corridor; no timing constraint.

    Endpoints follow the solver's task/goalset choice. Detaching the reference
    prevents this term from pulling the endpoints away from the requested goal.
    """
    start = position[:, :1].detach()
    end = position[:, -1:].detach()
    direction = end - start
    fraction = ((position - start) * direction).sum(-1)
    fraction = fraction / direction.square().sum(-1).clamp_min(1e-9)
    nearest = start + fraction.clamp(0., 1.)[..., None] * direction
    distance_squared = (position - nearest).square().sum(-1)
    return torch.clamp(distance_squared - radius**2, min=0.)


class VerticalToolPoseCost(ToolPoseCost):
    def __init__(self, config):
        super().__init__(config)
        self.vertical_weight = self.device_cfg.to_device([0.] * self.num_links)
        self.vertical_sign = self.device_cfg.to_device([1.] * self.num_links)
        self.vertical_cos_limit = self.device_cfg.to_device(
            [math.cos(math.radians(VERTICAL_TOLERANCE_DEG))] * self.num_links)
        self.vertical_channels = self.device_cfg.to_device([1.] + [0.] * (2*self.num_links-1))

    def setup_batch_tensors(self, batch_size, horizon, **kwargs):
        if self._horizon != horizon or self._batch_size != batch_size:
            self.vertical_time = torch.ones(horizon, device=self.device_cfg.device, dtype=self.device_cfg.dtype)
            self.vertical_time[0] = 0.; self.vertical_time[-1] = 0.
        super().setup_batch_tensors(batch_size, horizon, **kwargs)

    def forward(self, current_tool_poses, goal_tool_poses, idxs_goal=None, **kwargs):
        cost, position_error, rotation_error, index = super().forward(
            current_tool_poses, goal_tool_poses, idxs_goal, **kwargs)
        inclination = signed_axis_error(
            current_tool_poses.quaternion, self.vertical_sign, self.vertical_cos_limit)
        excursion = route_excursion(current_tool_poses.position)
        extra = ((inclination + ROUTE_WEIGHT_RATIO * excursion)
                 * self.vertical_weight).sum(-1) * self.vertical_time
        return cost + extra[..., None] * self.vertical_channels, position_error, rotation_error, index


def install_vertical_cost(motion):
    """Install before CUDA capture. Runtime switching only writes existing tensors."""
    core = motion.trajopt_solver.core
    rollouts = list(core.get_all_rollout_instances())
    rollouts += list(getattr(core,'additional_metrics_rollouts',{}).values())
    costs = []; seen = set()
    for rollout in rollouts:
        for name in ('cost_manager', 'metrics_cost_manager'):
            manager = getattr(rollout, name, None)
            if manager is None or id(manager) in seen:
                continue
            seen.add(id(manager))
            previous = manager.get_cost('tool_pose')
            if previous is None:
                continue
            cost = VerticalToolPoseCost(previous.config)
            if not previous.enabled:
                cost.disable_cost()
            if previous._batch_size > 0 and previous._horizon > 0:
                cost.setup_batch_tensors(previous._batch_size, previous._horizon)
            manager.costs['tool_pose'] = cost
            costs.append(cost)
    if not costs:
        raise RuntimeError('OPT 没有可接入的工具姿态代价')
    from curobo_ros_python.planning.curobo_adapter import set_vertical_costs

    set_vertical_costs(motion, costs)


@contextmanager
def vertical_tool_scope(motion, enabled, frames, goal_poses=None, goal_state=None):
    """Prefer the goal's up/down direction, then restore even if planning fails."""
    if not enabled:
        yield
        return
    if goal_poses is None:
        goal_poses = motion.compute_kinematics(goal_state).tool_poses
    from curobo_ros_python.planning.curobo_adapter import vertical_costs

    costs = vertical_costs(motion)
    saved = [(cost, cost.vertical_weight.clone(), cost.vertical_sign.clone()) for cost in costs]
    try:
        for frame in frames:
            quaternion = goal_poses.get_link_pose(frame).quaternion.reshape(-1, 4)[-1]
            z = 1. - 2. * (quaternion[1].square() + quaternion[2].square())
            sign = torch.where(z >= 0., 1., -1.)
            for cost in costs:
                index = cost.tool_frames.index(frame)
                cost.vertical_weight[index] = 10000.
                cost.vertical_sign[index] = sign
        yield
    finally:
        for cost, weight, sign in saved:
            cost.vertical_weight.copy_(weight)
            cost.vertical_sign.copy_(sign)
