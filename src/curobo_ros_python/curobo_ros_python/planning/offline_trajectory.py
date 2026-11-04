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
        raise ValueError('夹爪运动时间和采样周期必须是有限正数')
    result = JointTrajectory(joint_names=list(names))
    index = names.index(joint)
    start = np.array([positions[name] for name in names], dtype=np.float32)
    delta = target - float(start[index])
    samples = max(2, math.ceil(duration / dt) + 1)
    for t in np.linspace(0., duration, samples):
        u = t / duration
        q = start.copy()
        q[index] += delta * (10*u**3 - 15*u**4 + 6*u**5)
        velocity = np.zeros(len(names), dtype=np.float32)
        acceleration = velocity.copy()
        velocity[index] = delta * (30*u**2 - 60*u**3 + 30*u**4) / duration
        acceleration[index] = delta * (60*u - 180*u**2 + 120*u**3) / duration**2
        point = JointTrajectoryPoint(positions=q.tolist(), velocities=velocity.tolist(),
                                     accelerations=acceleration.tolist(),
                                     time_from_start=ros_duration(round(t*1e9)))
        result.points.append(point)
    return result


def concatenate_segments(segments):
    """Join matching joint columns at common boundaries; time starts at zero."""
    result = JointTrajectory()
    offset = 0
    for segment in segments:
        if not segment.points:
            raise ValueError('任务轨迹段不能为空')
        if not result.joint_names:
            result.joint_names = list(segment.joint_names)
            result.header = copy.deepcopy(segment.header)
        if list(segment.joint_names) != list(result.joint_names):
            raise ValueError('任务轨迹段的关节列不一致')
        if result.points and not np.allclose(result.points[-1].positions, segment.points[0].positions,
                                              atol=1e-3, rtol=0):
            raise ValueError('任务轨迹段边界位置不连续')
        first = duration_ns(segment.points[0].time_from_start)
        previous = -1
        for i, point in enumerate(segment.points):
            relative = duration_ns(point.time_from_start) - first
            if relative <= previous:
                raise ValueError('轨迹段的时间必须严格递增')
            previous = relative
            if result.points and i == 0:
                continue
            packed = copy.deepcopy(point)
            packed.time_from_start = ros_duration(offset + relative)
            result.points.append(packed)
        offset += previous
    return result


def stopped_segments(trajectory, boundaries, dt, limits, velocity_scale):
    """Retiming a fixed joint path to zero velocity/acceleration at stage ends."""
    import torch
    from curobo.types import JointState
    names=list(trajectory.joint_names)
    positions=trajectory.position.detach().cpu().numpy().reshape(-1,len(names))
    velocities=None if trajectory.velocity is None else trajectory.velocity.detach().cpu().numpy().reshape(positions.shape)
    if not np.isfinite(positions).all() or len(positions)<2:
        raise ValueError('联合轨迹位置无效')
    if boundaries[0]!=0 or any(b<=a for a,b in zip(boundaries,boundaries[1:])):
        raise ValueError('联合轨迹阶段时间无效')
    output_dt=float(trajectory.dt.reshape(-1)[0]) if trajectory.dt is not None else dt
    if boundaries[-1]>(len(positions)-1)*output_dt+1e-5:
        raise ValueError('阶段时间超出联合轨迹')
    def sample(indices):
        left=np.floor(indices).astype(int).clip(0,len(positions)-2)
        u=(indices-left)[:,None];p0=positions[left];p1=positions[left+1]
        if velocities is None: return p0+u*(p1-p0)
        return ((2*u**3-3*u**2+1)*p0+(u**3-2*u**2+u)*output_dt*velocities[left]
                +(-2*u**3+3*u**2)*p1+(u**3-u**2)*output_dt*velocities[left+1])
    result=[]
    for start,end in zip(boundaries,boundaries[1:]):
        duration=max(end-start,dt*2)
        for _ in range(8):
            count=max(17,math.ceil(duration/dt)+1)
            if count>10000: raise ValueError('联合轨迹重定时过长')
            u=np.linspace(0.,1.,count);warp=10*u**3-15*u**4+6*u**5
            q=sample((start+(end-start)*warp)/output_dt)
            v=np.gradient(q,dt,axis=0,edge_order=2);v[[0,-1]]=0.
            a=np.gradient(v,dt,axis=0,edge_order=2);a[[0,-1]]=0.
            jerk=np.gradient(a,dt,axis=0,edge_order=2)
            scale=max(1.,float(np.max(np.abs(v)*velocity_scale/limits[0])),
                float(np.sqrt(np.max(np.abs(a)*velocity_scale**2/limits[1]))),
                float(np.cbrt(np.max(np.abs(jerk)*velocity_scale**3/limits[2]))))
            if scale<=1.001: break
            duration=(count-1)*dt*scale*1.01
        else: raise ValueError('联合轨迹重定时未满足关节动态限位')
        if not all(np.isfinite(x).all() for x in [q,v,a,jerk]): raise ValueError('联合轨迹导数无效')
        tensor=lambda data:torch.as_tensor(data.copy(),device=trajectory.position.device,dtype=trajectory.position.dtype)
        js=JointState.from_position(tensor(q),joint_names=names)
        js.velocity=tensor(v);js.acceleration=tensor(a);js.jerk=tensor(jerk)
        result.append(js)
    return result
