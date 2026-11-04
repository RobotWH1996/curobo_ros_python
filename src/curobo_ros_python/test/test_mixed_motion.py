"""Targeted mixed-route and stage collision policy regressions (no robot I/O)."""
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock,patch
import numpy as np
import torch
from curobo_ros_interfaces.msg import MotionWaypoint,PickTask,PlaceTask
from curobo_ros_python.planning.groups import apply_robot
from curobo_ros_python.planning.pick_place_server import TaskBackend,PickPlaceServer
from curobo_ros_python.planning.pick_collision import contact_scope
from curobo_ros_python.planning.mesh_collision import MeshCollisionChecker


class MixedTargetTest(unittest.TestCase):
    def backend(self):
        apply_robot('lycus.yml')
        b=TaskBackend.__new__(TaskBackend)
        b.node=NS(_pose_in_base=lambda p:p);b.check=Mock()
        return b

    def test_mixed_order_and_joint_units_preserved(self):
        b=self.backend();task=PickTask(group_name='left_arm')
        joint=MotionWaypoint(type='joint',joint_positions=[.1]*7)
        pose=MotionWaypoint(type='pose');pose.pose.header.frame_id='base_link';pose.pose.pose.position.z=1.2
        task.waypoints=[joint,pose,joint]
        resolved=b._resolve_route(task)
        self.assertEqual([r.kind for r in resolved],['joint','pose','joint'])
        self.assertEqual(resolved[0].positions,[.1]*7)

    def test_bad_route_rejected(self):
        b=self.backend()
        for values in ([.1]*6,[float('nan')]*7):
            with self.assertRaises(ValueError): b._resolve_motion(MotionWaypoint(type='joint',joint_positions=values),'left_arm')
        with self.assertRaises(ValueError): b._resolve_motion(MotionWaypoint(),'left_arm')
        task=PickTask(group_name='left_arm')
        both=MotionWaypoint(type='joint',joint_positions=[.1]*7)
        both.pose.header.frame_id='base_link'
        task.waypoints=[both]
        with self.assertRaises(ValueError): b._resolve_route(task)

    def test_joint_and_pose_end_are_dispatched_outside_contact_segment(self):
        b=self.backend();b._move_segment=Mock()
        for task,attr,kind in ((PickTask(),'pick_end','joint'),(PlaceTask(),'place_end','pose')):
            target=NS(kind=kind);setattr(b,attr,target)
            b.end(task)
            b._move_segment.assert_called_with(task,'end',waypoint=target)
        self.assertIsNone(b._resolve_motion(MotionWaypoint(),'left_arm',allow_empty=True))


class CollisionPolicyTest(unittest.TestCase):
    def test_contact_scope_disables_only_two_fingers_and_optional_payload_then_restores(self):
        names={'left_gripper_left_jaw':0,'left_gripper_right_jaw':1,'pick_place_left_arm_payload':2,'arm':3}
        kin=NS(link_name_to_idx_map=names,link_spheres=torch.tensor([[[0.,0.,0.,.1]]*4]),
               get_sphere_index_from_link_name=lambda n:torch.tensor([names[n]]))
        payload=object();checker=NS(payload=payload,environment_ignore_links=set())
        planner=NS(_mesh_checker=checker);before=kin.link_spheres.clone()
        with patch('curobo_ros_python.planning.pick_collision.models',return_value=[kin]):
            with self.assertRaises(RuntimeError):
                with contact_scope(planner,'left_arm',True,disable_payload=True,suspend_fcl=True):
                    self.assertIsNone(checker.payload)
                    self.assertTrue(checker.skip_fcl_checks)
                    self.assertEqual(kin.link_spheres[0,:3,3].tolist(),[-100.]*3)
                    self.assertAlmostEqual(float(kin.link_spheres[0,3,3]),.1,places=6)
                    raise RuntimeError('cancel')
        self.assertIs(checker.payload,payload);self.assertFalse(checker.skip_fcl_checks)
        torch.testing.assert_close(kin.link_spheres,before)

    def test_fcl_gate_skips_native_queries_only_inside_scope(self):
        checker=MeshCollisionChecker.__new__(MeshCollisionChecker)
        checker._transforms=Mock(side_effect=RuntimeError('normal FCL entered'))
        checker.skip_fcl_checks=True
        self.assertIsNone(checker.check({}));checker._transforms.assert_not_called()
        checker.skip_fcl_checks=False
        with self.assertRaisesRegex(RuntimeError,'normal FCL entered'): checker.check({})

    def test_retreat_hides_released_box_and_points_and_restores_on_error(self):
        b=TaskBackend.__new__(TaskBackend);group='left_arm';box=object();scene=object();fcl_scene={}
        server=NS(held=None,placed={'part':box,'other':object()},contact_exclusions=[],preview_exclusions=[],invalidate_world=Mock())
        b.server=server;b.released_key='part';b.released_box=(np.eye(4),np.array([.1,.1,.1]))
        motion=NS(scene_collision_checker=NS(scene_model=scene),update_world=Mock())
        b.p=NS(_planners={group:motion},_mesh_scenes={group:fcl_scene})
        with self.assertRaises(RuntimeError):
            with b._released_object_scope(NS(group_name=group),'retreat'):
                self.assertNotIn('part',server.placed);self.assertIn('other',server.placed)
                points=np.array([[0,0,0],[1,0,0]],dtype=np.float32)
                kept=PickPlaceServer.filter_cloud(server,points)
                np.testing.assert_array_equal(kept,points[1:])
                raise RuntimeError('abort')
        self.assertIs(server.placed['part'],box);self.assertEqual(server.contact_exclusions,[])
        motion.update_world.assert_called_once_with(scene)


if __name__=='__main__': unittest.main()
