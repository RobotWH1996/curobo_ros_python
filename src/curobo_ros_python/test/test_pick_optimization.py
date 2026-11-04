"""Focused tests for path checks, stopped joints and per-task sensor snapshots."""
import time,unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock
import numpy as np
import torch
from geometry_msgs.msg import PoseStamped
from curobo.types import JointState
from curobo_ros_python.planning.pick_place_server import TaskBackend
from curobo_ros_python.planning.offline_trajectory import stopped_segments


class OptimizationTest(unittest.TestCase):
    def test_model_collection_includes_metrics_transition(self):
        from curobo_ros_python.planning.algorithms.trajopt import _robot_models
        def model():
            return NS(config=NS(kinematics_config=NS(fixed_transforms=object())))
        normal,metrics,extra=model(),model(),model()
        rollout=NS(transition_model=NS(robot_model=normal),metrics_transition_model=NS(robot_model=metrics))
        interpolated=NS(transition_model=NS(robot_model=extra),metrics_transition_model=NS(robot_model=metrics))
        core=NS(get_all_rollout_instances=Mock(return_value=[rollout]),additional_metrics_rollouts={'interpolated':interpolated})
        planner=NS(kinematics=normal,ik_solver=NS(core=core),trajopt_solver=NS(core=core),graph_planner=None)
        self.assertEqual([id(m) for m in _robot_models(planner)],[id(normal),id(metrics),id(extra)])

    def test_batch_line_retains_bend_reverse_and_orientation_checks(self):
        from curobo_ros_python.planning.cartesian_interpolation import validate_path
        path=np.array([[0.,0.,0.,1.,0.,0.,0.],[0.,0.,0.,1.,0.,0.,0.],[0.,0.,.05,1.,0.,0.,0.],[0.,0.,.1,1.,0.,0.,0.]],dtype=np.float32)
        validate_path(path,path[0],path[-1],.01,.05)
        for fault in ['bend','reverse','orientation','invalid']:
            changed=path.copy()
            if fault=='bend':changed[2,0]=.02
            elif fault=='reverse':changed[2,2]=-.02
            elif fault=='orientation':changed[2,3:]=[0.,1.,0.,0.]
            else:changed[2,0]=np.nan
            with self.assertRaises(RuntimeError,msg=fault):validate_path(changed,path[0],path[-1],.01,.05)

    def test_stopped_segments_preserve_boundary_and_dynamic_limits(self):
        q=torch.tensor([[0.,.3],[.05,.3],[.1,.3],[.05,.3],[0.,.3]],dtype=torch.float32)
        js=JointState.from_position(q,joint_names=['moving','fixed']);js.dt=torch.tensor([.02])
        limits=[np.array([1.,1.]),np.array([5.,5.]),np.array([50.,50.])]
        pieces=stopped_segments(js,[0.,.04,.08],.02,limits,.5)
        self.assertEqual(len(pieces),2)
        for piece in pieces:
            self.assertTrue(torch.equal(piece.velocity[[0,-1]],torch.zeros(2,2)))
            self.assertTrue(torch.equal(piece.acceleration[[0,-1]],torch.zeros(2,2)))
            np.testing.assert_allclose(piece.position[:,1],.3)
            self.assertLessEqual(float((piece.velocity.abs()*.5).max()),1.001)
            self.assertLessEqual(float((piece.acceleration.abs()*.25).max()),5.005)
            self.assertLessEqual(float((piece.jerk.abs()*.125).max()),50.05)
        torch.testing.assert_close(pieces[0].position[-1],pieces[1].position[0])
        torch.testing.assert_close(pieces[-1].position[-1],q[-1])

    def test_task_snapshot_reused_but_new_task_reads_new_cloud(self):
        node=NS(use_pointcloud=True,_pc_msg=object(),_pc_received=time.monotonic(),require_pointcloud=False,_p=lambda k:2.)
        calls=[]
        def sync(group,**kw):
            snapshot=kw.get('prepared_cloud') or (node._pc_msg,np.ones((1,3),dtype=np.float32),1)
            calls.append(snapshot);node._last_prepared_cloud=snapshot
        node._sync_world_from_pointcloud=sync
        def backend():
            b=object.__new__(TaskBackend);b.node=node;b._prepared_cloud=None;b._world_checked=False;b._changed_groups=set();b.check=Mock();return b
        first=backend();first._world('left_arm');old=calls[-1]
        node._pc_msg=object();first._world('left_arm');self.assertIs(calls[-1],old)
        second=backend();second._world('left_arm');self.assertIsNot(calls[-1],old)
        self.assertFalse(node.require_pointcloud)
