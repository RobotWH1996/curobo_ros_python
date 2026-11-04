"""Collision-unchecked Cartesian segments with continuous IK and bounded timing."""
from dataclasses import replace
import math
import numpy as np
import torch
from curobo.types import JointState, Pose, GoalToolPose
from curobo._src.solver.seed_ik.seed_ik_solver import SeedIKSolver
from curobo_ros_python.planning.offline_trajectory import stopped_segments


def sample_poses(start, end, fractions):
    """Linear position and shortest-arc quaternion interpolation, using wxyz."""
    fractions=np.asarray(fractions)
    qa=np.asarray(start[3:]);qa=qa/np.linalg.norm(qa)
    qb=np.asarray(end[3:]);qb=qb/np.linalg.norm(qb)
    dot=float(qa@qb)
    if dot<0: qb=-qb;dot=-dot
    if dot>.9995:
        quaternions=qa[None,:]+fractions[:,None]*(qb-qa)[None,:]
        quaternions/=np.linalg.norm(quaternions,axis=1)[:,None]
    else:
        theta=math.acos(min(1.,dot))
        quaternions=(np.sin((1-fractions)*theta)[:,None]*qa[None,:]
                     +np.sin(fractions*theta)[:,None]*qb[None,:])/math.sin(theta)
    positions=np.asarray(start[:3])[None,:]+fractions[:,None]*(np.asarray(end[:3])-start[:3])[None,:]
    return np.concatenate((positions,quaternions),axis=1)


def validate_path(actual, start, end, position_tolerance, orientation_tolerance):
    """Check line, progress and interpolated orientation after joint retiming."""
    if not np.isfinite(actual).all(): raise RuntimeError('插补 FK 含无效数值')
    delta=np.asarray(end[:3])-start[:3];length=float(np.linalg.norm(delta))
    if length<1e-6: raise RuntimeError('插补段平移距离太小')
    direction=delta/length;offset=actual[:,:3]-start[:3];along=offset@direction
    perpendicular=np.linalg.norm(offset-along[:,None]*direction,axis=1)
    if (np.max(perpendicular)>position_tolerance or np.min(along)<-position_tolerance
            or np.max(along)>length+position_tolerance or np.min(np.diff(along))<-position_tolerance):
        raise RuntimeError('插补轨迹偏离直线或出现反向')
    expected=sample_poses(start,end,np.clip(along/length,0,1))
    q=actual[:,3:];q=q/np.linalg.norm(q,axis=1)[:,None]
    angle=2*np.arccos(np.clip(np.abs((q*expected[:,3:]).sum(1)),0,1))
    if np.max(angle)>orientation_tolerance+.005: raise RuntimeError('插补轨迹姿态偏离 SLERP 路径')
    if np.linalg.norm(actual[-1,:3]-end[:3])>position_tolerance: raise RuntimeError('插补终点未到达目标')


def kinematic_solver(motion,key,num_seeds=1):
    """Cache an IK-only solver; copy locked transforms without collision state."""
    cache=getattr(motion,'_contact_ik_cache',None)
    if cache is None: cache={};motion._contact_ik_cache=cache
    if key not in cache:
        cache[key]=SeedIKSolver(replace(motion.ik_solver.seed_ik_solver.config,num_seeds=num_seeds))
    instance=cache[key];reference=motion.ik_solver.kinematics
    for model in (instance._robot_model,instance.kinematics):
        if list(model.joint_names)!=list(reference.joint_names): raise RuntimeError('插补 IK 关节配置不一致')
        model.config.kinematics_config.fixed_transforms.copy_(reference.config.kinematics_config.fixed_transforms)
        if model.lock_jointstate is not None and reference.lock_jointstate is not None:
            model.lock_jointstate.position.copy_(reference.lock_jointstate.position)
    return instance


def refine_ik(planner,motion,goal,previous):
    """Pose-only fallback for LM singularities; scene is empty and self checks off."""
    from curobo.inverse_kinematics import InverseKinematics,InverseKinematicsCfg
    solver=getattr(motion,'_contact_refine_ik',None)
    if solver is None:
        config=InverseKinematicsCfg.create(robot=motion.ik_solver.config.robot_config,
            self_collision_check=False,num_seeds=4,use_cuda_graph=True,
            position_tolerance=planner._position_tolerance,orientation_tolerance=planner._orientation_tolerance)
        config.use_lm_seed=False
        solver=InverseKinematics(config);motion._contact_refine_ik=solver
    reference=motion.ik_solver.kinematics
    for rollout in solver.core.get_all_rollout_instances():
        for name in ('transition_model','metrics_transition_model'):
            transition=getattr(rollout,name,None)
            if transition is None:continue
            model=transition.robot_model
            model.config.kinematics_config.fixed_transforms.copy_(reference.config.kinematics_config.fixed_transforms)
            if model.lock_jointstate is not None and reference.lock_jointstate is not None:
                model.lock_jointstate.position.copy_(reference.lock_jointstate.position)
    result=solver.solve_pose(goal,seed_config=previous.reshape(1,1,-1).repeat(1,4,1),return_seeds=4)
    valid=result.success.reshape(-1);q=result.solution.reshape(-1,previous.numel())
    if not bool(valid.any()):return None
    q=q[valid];return q[torch.argmin((q-previous.reshape(1,-1)).square().sum(-1))].detach().clone()


def interpolate_cartesian(planner,motion,current,target,tool,dt,velocity_scale,check):
    """Use one local LM seed per sample; never query scene or self collisions."""
    from curobo_ros_python.planning.convert import stamped_to_pose
    start_pose=planner.fk_tool_poses(current)[tool]
    start=np.r_[start_pose.position.detach().cpu().numpy().reshape(-1,3)[0],
                start_pose.quaternion.detach().cpu().numpy().reshape(-1,4)[0]]
    p=target.pose.position;q=target.pose.orientation
    end=np.array([p.x,p.y,p.z,q.w,q.x,q.y,q.z])
    if not np.isfinite(end).all() or np.linalg.norm(end[3:])<1e-8: raise ValueError('插补目标位姿无效')
    angle=2*math.acos(min(1.,abs(float((start[3:]/np.linalg.norm(start[3:]))@(end[3:]/np.linalg.norm(end[3:]))))))
    count=max(2,math.ceil(np.linalg.norm(end[:3]-start[:3])/.005),math.ceil(angle/.03))+1
    seed=kinematic_solver(motion,'endpoint');names=list(seed.joint_names)
    first=motion.kinematics.get_active_js(current).position.reshape(1,-1).clone()
    fixed_fk=planner.fk_tool_poses(current)
    endpoint_goal,_=planner._goal_for_ee_poses(current,{tool:stamped_to_pose(target,planner.device_cfg)},current_fk=fixed_fk)
    endpoint=seed.solve_single(endpoint_goal,seed_config=first.reshape(1,1,-1),return_seeds=1)
    check();rows=None;iterations=1
    endpoint_q=endpoint.solution.reshape(1,-1).detach().clone() if bool(endpoint.success.all()) else None
    if endpoint_q is None:
        refined=refine_ik(planner,motion,endpoint_goal,first)
        if refined is not None:endpoint_q=refined.reshape(1,-1)
    if endpoint_q is not None:
        bucket=next(n for n in (33,65,129,257,513) if n>=count)
        batch_solver=kinematic_solver(motion,bucket);fractions=np.linspace(0.,1.,bucket)
        desired=sample_poses(start,end,fractions);pose_map={}
        for frame in motion.tool_frames:
            if frame==tool:
                pose_map[frame]=Pose(position=planner.device_cfg.to_device(desired[:,:3]),
                    quaternion=planner.device_cfg.to_device(desired[:,3:]),normalize_rotation=False)
            else:
                fixed=fixed_fk[frame]
                pose_map[frame]=Pose(position=fixed.position.reshape(1,3).repeat(bucket,1),
                    quaternion=fixed.quaternion.reshape(1,4).repeat(bucket,1),normalize_rotation=False)
        goal=GoalToolPose.from_poses(pose_map,ordered_tool_frames=motion.tool_frames,num_goalset=1)
        last=endpoint_q
        alpha=planner.device_cfg.to_device(fractions).reshape(-1,1)
        result=batch_solver.solve_batch(goal,seed_config=(first+alpha*(last-first)).unsqueeze(1),return_seeds=1);iterations+=1
        candidates=result.solution.reshape(bucket,len(names)).detach().clone()
        if (bool(result.success.all()) and bool(torch.isfinite(candidates).all())
                and float(torch.diff(candidates,dim=0).abs().max())<=.15):
            candidates[0]=first.reshape(-1);rows=list(candidates.unbind(0))
    if rows is None:
        # Near a singularity use local continuation with adaptive bisection.
        previous=first;rows=[first.reshape(-1).clone()];alphas=[0.]
        pending=list(np.linspace(0.,1.,count)[1:])
        while pending:
            check();alpha=pending.pop(0);desired=sample_poses(start,end,[alpha])[0]
            pose=Pose(position=planner.device_cfg.to_device(desired[:3]).reshape(1,3),
                      quaternion=planner.device_cfg.to_device(desired[3:]).reshape(1,4),normalize_rotation=False)
            goal,_=planner._goal_for_ee_poses(current,{tool:pose},current_fk=fixed_fk)
            result=seed.solve_single(goal,seed_config=previous.reshape(1,1,-1),return_seeds=1);iterations+=1
            candidate=result.solution.reshape(-1,len(names))[0].detach().clone()
            good=bool(result.success.any())
            jump=float((candidate-previous.reshape(-1)).abs().max())
            if not good or jump>.15:
                refined=refine_ik(planner,motion,goal,previous)
                if refined is not None:
                    candidate=refined;good=True;jump=float((candidate-previous.reshape(-1)).abs().max())
            if not good or not math.isfinite(jump) or jump>.15:
                if alpha-alphas[-1]>1e-4 and iterations<2000:
                    pending[:0]=[(alpha+alphas[-1])*.5,alpha];continue
                raise RuntimeError(f'笛卡尔插补 IK 不连续或不可达，进度={alpha:.4f}，最大关节变化={jump:.3f}rad')
            rows.append(candidate);alphas.append(alpha);previous=candidate.reshape(1,-1)
    path=JointState.from_position(torch.stack(rows),joint_names=names)
    path.dt=torch.as_tensor([dt],device=path.position.device,dtype=path.position.dtype)
    path.velocity=torch.as_tensor(np.gradient(path.position.cpu().numpy(),dt,axis=0,edge_order=2),device=path.position.device,dtype=path.position.dtype)
    transition=motion.trajopt_solver.transition_model
    limits=[x.detach().cpu().numpy().reshape(-1).clip(1e-6) for x in
            (transition.max_velocity,transition.max_acceleration,transition.max_jerk)]
    trajectory=stopped_segments(path,[0.,(len(rows)-1)*dt],dt,limits,velocity_scale)[0]
    bounds=motion.kinematics.get_joint_limits().position
    if not bool(((trajectory.position>=bounds[0])&(trajectory.position<=bounds[1])).all()):
        raise RuntimeError('插补重定时后的轨迹超过关节限位')
    actual=planner.fk_tool_path(trajectory,tool)
    validate_path(actual,start,end,planner._position_tolerance,planner._orientation_tolerance)
    check()
    return trajectory,dict(ik_samples=len(rows),ik_calls=iterations,points=len(actual))
