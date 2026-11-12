"""Export and stitch offline task trajectories without issuing commands."""

import copy
import math

import numpy as np
from builtin_interfaces.msg import Duration
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


def duration_ns(duration):
    return duration.sec * 1_000_000_000 + duration.nanosec


def ros_duration(nanoseconds):
    seconds, remainder = divmod(int(nanoseconds), 1_000_000_000)
    return Duration(sec=seconds, nanosec=remainder)


def gripper_segment(names, positions, joint, target, duration, dt):
    """Quintic joint motion, holding every other robot joint stationary."""
    if not math.isfinite(duration) or duration <= 0 or not math.isfinite(dt) or dt <= 0:
        raise ValueError("夹爪运动时间和采样周期必须是有限正数")
    result = JointTrajectory(joint_names=list(names))
    index = names.index(joint)
    start = np.array([positions[name] for name in names], dtype=np.float32)
    delta = target - float(start[index])
    samples = max(2, math.ceil(duration / dt) + 1)
    for t in np.linspace(0.0, duration, samples):
        u = t / duration
        q = start.copy()
        q[index] += delta * (10 * u**3 - 15 * u**4 + 6 * u**5)
        velocity = np.zeros(len(names), dtype=np.float32)
        acceleration = velocity.copy()
        velocity[index] = delta * (30 * u**2 - 60 * u**3 + 30 * u**4) / duration
        acceleration[index] = delta * (60 * u - 180 * u**2 + 120 * u**3) / duration**2
        point = JointTrajectoryPoint(
            positions=q.tolist(),
            velocities=velocity.tolist(),
            accelerations=acceleration.tolist(),
            time_from_start=ros_duration(round(t * 1e9)),
        )
        result.points.append(point)
    return result


def concatenate_segments(segments):
    """Join matching joint columns at common boundaries; time starts at zero."""
    result = JointTrajectory()
    offset = 0
    for segment in segments:
        if not segment.points:
            raise ValueError("任务轨迹段不能为空")
        if not result.joint_names:
            result.joint_names = list(segment.joint_names)
            result.header = copy.deepcopy(segment.header)
        if list(segment.joint_names) != list(result.joint_names):
            raise ValueError("任务轨迹段的关节列不一致")
        if result.points and not np.allclose(
            result.points[-1].positions, segment.points[0].positions, atol=1e-3, rtol=0
        ):
            raise ValueError("任务轨迹段边界位置不连续")
        first = duration_ns(segment.points[0].time_from_start)
        previous = -1
        for i, point in enumerate(segment.points):
            relative = duration_ns(point.time_from_start) - first
            if relative <= previous:
                raise ValueError("轨迹段的时间必须严格递增")
            previous = relative
            if result.points and i == 0:
                continue
            packed = copy.deepcopy(point)
            packed.time_from_start = ros_duration(offset + relative)
            result.points.append(packed)
        offset += previous
    return result


def cpu_dynamic_limits(transition):
    """Copy velocity, acceleration and jerk limits to the host once."""
    from curobo_ros_python.planning.curobo_adapter import (
        host_dynamic_limits,
        set_host_dynamic_limits,
    )

    cached = host_dynamic_limits(transition)
    if cached is not None:
        return cached
    import torch

    rows = torch.stack(
        [
            item.detach().reshape(-1)
            for item in (transition.max_velocity, transition.max_acceleration, transition.max_jerk)
        ]
    )
    cached = [row.clip(1e-6) for row in rows.cpu().numpy()]
    set_host_dynamic_limits(transition, cached)
    return cached


def host_state_arrays(position, velocity=None, acceleration=None):
    """Download matching trajectory tensors in one GPU sync."""
    import torch

    def flat(tensor):
        return tensor.detach().reshape(-1, tensor.shape[-1])

    parts = [flat(position)]
    if velocity is not None:
        parts.append(flat(velocity))
    if acceleration is not None:
        parts.append(flat(acceleration))
    host = torch.stack(parts).cpu().numpy()
    return host if len(parts) > 1 else host[0]


def stopped_segments(
    trajectory, boundaries, dt, limits, velocity_scale, positions=None, velocities=None
):
    """Retiming a fixed joint path to zero velocity/acceleration at stage ends."""
    import torch
    from curobo.types import JointState

    names = list(trajectory.joint_names)
    if positions is None:
        positions = trajectory.position.detach().cpu().numpy().reshape(-1, len(names))
    else:
        positions = np.asarray(positions).reshape(-1, len(names))
    if velocities is None and trajectory.velocity is not None:
        velocities = trajectory.velocity.detach().cpu().numpy().reshape(positions.shape)
    elif velocities is not None:
        velocities = np.asarray(velocities).reshape(positions.shape)
    if not np.isfinite(positions).all() or len(positions) < 2:
        raise ValueError("联合轨迹位置无效")
    if boundaries[0] != 0 or any(b <= a for a, b in zip(boundaries, boundaries[1:])):
        raise ValueError("联合轨迹阶段时间无效")
    output_dt = float(trajectory.dt.reshape(-1)[0]) if trajectory.dt is not None else dt
    if boundaries[-1] > (len(positions) - 1) * output_dt + 1e-5:
        raise ValueError("阶段时间超出联合轨迹")

    def sample(indices):
        left = np.floor(indices).astype(int).clip(0, len(positions) - 2)
        u = (indices - left)[:, None]
        p0 = positions[left]
        p1 = positions[left + 1]
        if velocities is None:
            return p0 + u * (p1 - p0)
        return (
            (2 * u**3 - 3 * u**2 + 1) * p0
            + (u**3 - 2 * u**2 + u) * output_dt * velocities[left]
            + (-2 * u**3 + 3 * u**2) * p1
            + (u**3 - u**2) * output_dt * velocities[left + 1]
        )

    result = []
    for start, end in zip(boundaries, boundaries[1:]):
        duration = max(end - start, dt * 2)
        for _ in range(8):
            count = max(17, math.ceil(duration / dt) + 1)
            if count > 10000:
                raise ValueError("联合轨迹重定时过长")
            u = np.linspace(0.0, 1.0, count)
            warp = 10 * u**3 - 15 * u**4 + 6 * u**5
            q = sample((start + (end - start) * warp) / output_dt)
            v = np.gradient(q, dt, axis=0, edge_order=2)
            v[[0, -1]] = 0.0
            a = np.gradient(v, dt, axis=0, edge_order=2)
            a[[0, -1]] = 0.0
            jerk = np.gradient(a, dt, axis=0, edge_order=2)
            scale = max(
                1.0,
                float(np.max(np.abs(v) * velocity_scale / limits[0])),
                float(np.sqrt(np.max(np.abs(a) * velocity_scale**2 / limits[1]))),
                float(np.cbrt(np.max(np.abs(jerk) * velocity_scale**3 / limits[2]))),
            )
            if scale <= 1.001:
                break
            duration = (count - 1) * dt * scale * 1.01
        else:
            raise ValueError("联合轨迹重定时未满足关节动态限位")
        if not all(np.isfinite(x).all() for x in [q, v, a, jerk]):
            raise ValueError("联合轨迹导数无效")
        gpu = torch.as_tensor(
            np.stack((q, v, a, jerk)),
            device=trajectory.position.device,
            dtype=trajectory.position.dtype,
        )
        js = JointState.from_position(gpu[0], joint_names=names)
        js.velocity = gpu[1]
        js.acceleration = gpu[2]
        js.jerk = gpu[3]
        result.append(js)
    return result


GRIPPER_STAGES = frozenset(("open", "close", "release"))


def _point_seconds(point):
    return point.time_from_start.sec + point.time_from_start.nanosec * 1e-9


def overlay_joint_motion(trajectory, joint, start, target, until):
    """Quintic one joint from start to target over ``until`` seconds, then hold the target."""
    if until <= 0 or not math.isfinite(until):
        raise ValueError("夹爪张开时间必须是有限正数")
    # Put the opening completion exactly on a knot, then hold the target.
    trajectory = copy.deepcopy(trajectory)
    times = np.array([_point_seconds(point) for point in trajectory.points])
    if times[0] < until < times[-1] and np.min(np.abs(times - until)) > 1e-8:
        _, q, v, a, _ = sample_trajectory(trajectory, sample_times=[until])
        point = JointTrajectoryPoint(
            positions=q[0].tolist(),
            velocities=v[0].tolist(),
            accelerations=a[0].tolist(),
            time_from_start=ros_duration(round(until * 1e9)),
        )
        trajectory.points.insert(int(np.searchsorted(times, until)), point)
    index = trajectory.joint_names.index(joint)
    start = float(start)
    delta = float(target) - start
    result = JointTrajectory()
    result.joint_names = list(trajectory.joint_names)
    result.header = copy.deepcopy(trajectory.header)
    for point in trajectory.points:
        copied = copy.deepcopy(point)
        elapsed = _point_seconds(copied)
        u = min(max(elapsed / until, 0.0), 1.0)
        shape = 10 * u**3 - 15 * u**4 + 6 * u**5
        copied.positions[index] = start + delta * shape
        if copied.velocities:
            speed = 0.0 if elapsed >= until else delta * (30 * u**2 - 60 * u**3 + 30 * u**4) / until
            copied.velocities[index] = speed
        if copied.accelerations:
            accel = (
                0.0 if elapsed >= until else delta * (60 * u - 180 * u**2 + 120 * u**3) / until**2
            )
            copied.accelerations[index] = accel
        result.points.append(copied)
    return result


def continuous_runs(stages, gripper_stages=GRIPPER_STAGES):
    """Split a task into gripper holds and the arm motions between them.

    Arm motions inside one run are retimed together. A run ends at a gripper
    stage, so the grasp and place contacts stay stopped.
    """
    runs = []
    start = 0
    for index, name in enumerate(stages):
        if name not in gripper_stages:
            continue
        if start < index:
            runs.append(("motion", start, index))
        runs.append(("gripper", index, index + 1))
        start = index + 1
    if start < len(stages):
        runs.append(("motion", start, len(stages)))
    return runs


def sample_trajectory(trajectory, dt=0.005, sample_times=None):
    """Sample the controller's position/velocity/acceleration quintic, not just knots."""
    times = np.array([_point_seconds(point) for point in trajectory.points])
    if len(times) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError("轨迹必须至少有两个时间严格递增的点")
    points = trajectory.points
    q = np.array([point.positions for point in points], dtype=float)
    if not np.isfinite(q).all():
        raise ValueError("轨迹位置含非有限值")
    have_v = all(len(point.velocities) == q.shape[1] for point in points)
    have_a = have_v and all(len(point.accelerations) == q.shape[1] for point in points)
    velocities = np.array([point.velocities for point in points]) if have_v else None
    accelerations = np.array([point.accelerations for point in points]) if have_a else None
    return _sample_trajectory_arrays(times, q, velocities, accelerations, dt, sample_times)


def _sample_trajectory_arrays(
    times, q, velocities=None, accelerations=None, dt=0.005, sample_times=None
):
    """Evaluate the same controller curve directly from time and joint arrays."""
    have_v = velocities is not None
    have_a = have_v and accelerations is not None
    h = np.diff(times)[:, None]
    if len(times) < 2 or np.any(h <= 0):
        raise ValueError("轨迹必须至少有两个时间严格递增的点")
    coefficients = np.zeros((6, len(times) - 1, q.shape[1]))
    coefficients[0] = q[:-1]
    if have_v:
        v = velocities
        coefficients[1] = v[:-1] * h
        if have_a:
            a = accelerations
            coefficients[2] = 0.5 * a[:-1] * h * h
            c0 = q[1:] - coefficients[:3].sum(0)
            c1 = v[1:] * h - coefficients[1] - 2 * coefficients[2]
            c2 = a[1:] * h * h - 2 * coefficients[2]
            coefficients[3] = 10 * c0 - 4 * c1 + 0.5 * c2
            coefficients[4] = -15 * c0 + 7 * c1 - c2
            coefficients[5] = 6 * c0 - 3 * c1 + 0.5 * c2
        else:
            delta = q[1:] - q[:-1]
            coefficients[2] = 3 * delta - (2 * v[:-1] + v[1:]) * h
            coefficients[3] = -2 * delta + (v[:-1] + v[1:]) * h
    else:
        coefficients[1] = np.diff(q, axis=0)
    ts = (
        np.asarray(sample_times, dtype=float)
        if sample_times is not None
        else np.unique(
            np.concatenate(
                [
                    times,
                    np.linspace(
                        times[0], times[-1], max(2, math.ceil((times[-1] - times[0]) / dt) + 1)
                    ),
                ]
            )
        )
    )
    index = np.searchsorted(times, ts, side="right").clip(1, len(times) - 1) - 1
    u = ((ts - times[index]) / h[index, 0])[:, None]
    values = []
    for derivative in range(4):
        value = np.zeros((len(ts), q.shape[1]))
        for power in range(derivative, 6):
            multiplier = math.factorial(power) / math.factorial(power - derivative)
            value += multiplier * coefficients[power, index] * u ** (power - derivative)
        values.append(value / h[index] ** derivative)
    if not all(np.isfinite(value).all() for value in values):
        raise ValueError("轨迹导数含非有限值")
    return ts, *values


# Blend before deep braking; keep the global velocity scale.
_THROUGH_FRACTION = 0.45


def _knot_trajectory(joint_names, positions, velocities, accelerations, times):
    trajectory = JointTrajectory(joint_names=list(joint_names))
    for q, dq, ddq, t in zip(positions, velocities, accelerations, times):
        trajectory.points.append(
            JointTrajectoryPoint(
                positions=q.tolist(),
                velocities=dq.tolist(),
                accelerations=ddq.tolist(),
                time_from_start=ros_duration(round(t * 1e9)),
            )
        )
    return trajectory


def _stage_source(paths, joint_names, dt, timings, velocities, accelerations):
    """Join the stage paths on their source time axes, stopped at every boundary."""
    segments = []
    boundaries = [0.0]
    for index, path in enumerate(paths):
        path = np.asarray(path, dtype=float)
        ts = (
            np.arange(len(path)) * dt
            if timings is None
            else np.asarray(timings[index], dtype=float)
        )
        if len(path) < 2 or np.any(np.diff(ts) <= 0) or not np.isfinite(path).all():
            raise ValueError("连续轨迹的点或原始时间轴无效")
        ts = ts - ts[0]
        v = (
            np.gradient(path, ts, axis=0)
            if velocities is None
            else np.asarray(velocities[index], dtype=float)
        )
        a = (
            np.gradient(v, ts, axis=0)
            if accelerations is None
            else np.asarray(accelerations[index], dtype=float)
        )
        v = v.copy()
        a = a.copy()
        v[[0, -1]] = 0.0
        a[[0, -1]] = 0.0
        segments.append(_knot_trajectory(joint_names, path, v, a, ts))
        boundaries.append(boundaries[-1] + ts[-1])
    return concatenate_segments(segments), boundaries


def _junction_span(source_times, source_speed, boundaries, index):
    """Where motion is still fast on each side of one stage boundary, or None."""
    center = boundaries[index]
    left_range = np.flatnonzero((source_times >= boundaries[index - 1]) & (source_times < center))
    right_range = np.flatnonzero((source_times > center) & (source_times <= boundaries[index + 1]))
    if not len(left_range) or not len(right_range):
        return None
    left_moving = left_range[
        source_speed[left_range] >= max(_THROUGH_FRACTION * source_speed[left_range].max(), 1e-5)
    ]
    right_moving = right_range[
        source_speed[right_range] >= max(_THROUGH_FRACTION * source_speed[right_range].max(), 1e-5)
    ]
    if not len(left_moving) or not len(right_moving):
        return None
    return float(source_times[left_moving[-1]]), float(source_times[right_moving[0]])


def _through_knots(source, left, center, right, index, contacts, dt):
    """Knots that cross one boundary at speed, or None when it has to stay a stop."""
    _, q, v, a, _ = sample_trajectory(source, sample_times=np.array([left, center, right]))
    incoming = q[1] - q[0]
    outgoing = q[2] - q[1]
    dl = float(np.linalg.norm(incoming))
    dr = float(np.linalg.norm(outgoing))
    if min(dl, dr) < 1e-7:
        return None
    incoming /= dl
    outgoing /= dr
    pace = min(float(np.linalg.norm(v[0])), float(np.linalg.norm(v[2])))
    if contacts is not None and contacts[index] and not contacts[index - 1]:
        direction = outgoing
    elif contacts is not None and contacts[index - 1] and not contacts[index]:
        direction = incoming
    else:
        direction = incoming + outgoing
    norm = float(np.linalg.norm(direction))
    # An exact geometric reversal cannot have nonzero through velocity.
    if norm < 1e-4 or float(incoming @ outgoing) < -0.98:
        return None
    v[1] = direction / norm * pace
    a[1] = 0.0
    left_duration = min(
        center - left, max(dt, 1.1 * dl / max(0.5 * (np.linalg.norm(v[0]) + pace), 1e-6))
    )
    right_duration = min(
        right - center, max(dt, 1.1 * dr / max(0.5 * (np.linalg.norm(v[2]) + pace), 1e-6))
    )
    return q, v, a, left_duration, right_duration


def _junction_windows(source, source_times, boundaries, joint_names, dt, stops, contacts):
    """Short replacement windows for the boundaries that can be crossed without stopping."""
    # Find the actual moving tails, not a fixed 250 ms window in padded stops.
    source_speed = np.linalg.norm(np.array([p.velocities for p in source.points]), axis=1)
    windows = []
    shift = 0.0
    for index in range(1, len(boundaries) - 1):
        if index in stops:
            continue
        span = _junction_span(source_times, source_speed, boundaries, index)
        if span is None:
            continue
        left, right = span
        if windows and left < windows[-1][1] - 1e-8:
            continue
        center = boundaries[index]
        knots = _through_knots(source, left, center, right, index, contacts, dt)
        if knots is None:
            continue
        q, v, a, left_duration, right_duration = knots
        old = np.array([left, center, right])
        new = np.array(
            [
                left + shift,
                left + shift + left_duration,
                left + shift + left_duration + right_duration,
            ]
        )
        window = _knot_trajectory(joint_names, q, v, a, new)
        windows.append((left, right, window, old, new))
        shift += left_duration + right_duration - (right - left)
    return windows


def _merge_windows(source, source_times, windows):
    """Sample the source on every knot, then substitute the blended windows."""
    extra = [t for _, _, _, old, _ in windows for t in old]
    original_times = np.unique(np.round(np.r_[source_times, extra], 9))
    times = original_times.copy()
    _, q, v, a, _ = sample_trajectory(source, sample_times=original_times)
    for left, right, window, old, new in windows:
        mask = (original_times >= left - 1e-9) & (original_times <= right + 1e-9)
        mapped = np.interp(original_times[mask], old, new)
        times[mask] = mapped
        times[original_times > right + 1e-9] += (new[-1] - new[0]) - (right - left)
        _, q[mask], v[mask], a[mask], _ = sample_trajectory(window, sample_times=mapped)
    return original_times, times, q, v, a


def _slowdown_clock(samples, bumps):
    """Warped time, local time scale and its slope for Gaussian slow-down bumps."""
    from scipy.special import erf

    samples = np.asarray(samples, dtype=float)
    mapped = samples.copy()
    scale = np.ones_like(samples)
    slope = np.zeros_like(samples)
    for center, width, amount in bumps:
        z = (samples - center) / width
        weight = amount * np.exp(-0.5 * z * z)
        scale += weight
        slope -= weight * z / width
        mapped += (
            amount
            * width
            * np.sqrt(np.pi / 2)
            * (erf(z / np.sqrt(2)) - erf(-center / width / np.sqrt(2)))
        )
    return mapped, scale, slope


def _limit_ratios(retimed, output_times, q, velocity, acceleration, limits, velocity_scale):
    """Per-interval factor by which the exported quintic exceeds the dynamic limits."""
    spans = np.diff(retimed)
    check_times = (retimed[:-1, None] + spans[:, None] * np.linspace(0, 1, 9)).reshape(-1)
    _, _, dv, da, dj = _sample_trajectory_arrays(
        output_times, q, velocity, acceleration, sample_times=check_times
    )
    ratio = np.ones(len(spans))
    for order, values, limit in zip((1, 2, 3), (dv, da, dj), limits):
        peak = np.max(
            (np.abs(values) * velocity_scale**order / limit).reshape(len(spans), 9, -1),
            axis=(1, 2),
        )
        ratio = np.maximum(ratio, np.where(peak > 1.005, peak ** (1.0 / order) * 1.02, 1.0))
    return ratio


def _retime_within_limits(times, q, v, a, dt, limits, velocity_scale):
    """Slow down only the intervals that exceed dynamic limits.

    Smooth local time warps, including the chain-rule acceleration term.
    Independent interval scaling creates new jumps and spreads slowdown.
    """
    bumps = []
    for _ in range(24):
        retimed, node_scale, node_slope = _slowdown_clock(times, bumps)
        # Match the nanosecond rounding of the exported ROS Duration exactly.
        nanoseconds = np.rint(retimed * 1e9).astype(np.int64)
        output_times = nanoseconds // 1_000_000_000 + (nanoseconds % 1_000_000_000) * 1e-9
        scaled_velocity = v / node_scale[:, None]
        scaled_acceleration = (
            a / node_scale[:, None] ** 2 - v * node_slope[:, None] / node_scale[:, None] ** 3
        )
        ratio = _limit_ratios(
            retimed, output_times, q, scaled_velocity, scaled_acceleration, limits, velocity_scale
        )
        if ratio.max() <= 1.001:
            break
        worst = int(np.argmax(ratio))
        center = float((times[worst] + times[worst + 1]) * 0.5)
        _, local_scale, _ = _slowdown_clock([center], bumps)
        amount = float((ratio[worst] - 1.0) * local_scale[0] * 1.08)
        bumps.append(
            (center, max(0.2 * velocity_scale, 2 * (times[worst + 1] - times[worst])), amount)
        )
        if retimed[-1] > 10000 * dt:
            raise ValueError("局部重定时过长")
    else:
        raise ValueError("衔接局部重定时未满足动态限位")
    return output_times, scaled_velocity, scaled_acceleration


def blend_through(
    paths,
    joint_names,
    dt,
    limits,
    velocity_scale,
    step=0.02,
    metadata=False,
    timings=None,
    velocities=None,
    accelerations=None,
    stops=(),
    contacts=None,
):
    """Keep source timing and interpolation; blend only short junction windows.

    Only intervals exceeding dynamic limits are stretched. No duration is
    inferred from joint path length. Input/output times precede velocity_scale.
    """
    import torch
    from curobo.types import JointState

    source, boundaries = _stage_source(paths, joint_names, dt, timings, velocities, accelerations)
    source_times = np.array([_point_seconds(p) for p in source.points])
    windows = _junction_windows(source, source_times, boundaries, joint_names, dt, stops, contacts)
    original_times, times, q, v, a = _merge_windows(source, source_times, windows)
    output_times, scaled_velocity, scaled_acceleration = _retime_within_limits(
        times, q, v, a, dt, limits, velocity_scale
    )
    _, position, velocity, acceleration, jerk = _sample_trajectory_arrays(
        output_times, q, scaled_velocity, scaled_acceleration, sample_times=output_times
    )
    tensor = lambda values: torch.as_tensor(values.copy(), dtype=torch.float64)
    state = JointState.from_position(tensor(position), joint_names=list(joint_names))
    state.velocity = tensor(velocity)
    state.acceleration = tensor(acceleration)
    state.jerk = tensor(jerk)
    state.dt = tensor([dt])
    phase_times = np.interp(boundaries, original_times, output_times)
    return (state, output_times, phase_times) if metadata else state


def insert_stationary_hold(trajectory, at, duration):
    """Insert planned arm hold at an existing stopped boundary; no feedback wait."""
    result = copy.deepcopy(trajectory)
    index = min(range(len(result.points)), key=lambda i: abs(_point_seconds(result.points[i]) - at))
    if abs(_point_seconds(result.points[index]) - at) > 1e-6:
        raise ValueError("停留位置不在阶段边界")
    point = result.points[index]
    if max([abs(x) for x in point.velocities + point.accelerations] or [0.0]) > 1e-5:
        raise ValueError("夹爪等待点必须保持手臂静止")
    for following in result.points[index + 1 :]:
        following.time_from_start = ros_duration(
            round((_point_seconds(following) + duration) * 1e9)
        )
    held = copy.deepcopy(point)
    held.time_from_start = ros_duration(round((at + duration) * 1e9))
    result.points.insert(index + 1, held)
    return result
