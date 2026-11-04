"""Offline pick/place sequencing, feedback and conservative geometry tests."""
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock
import numpy as np

from curobo_ros_python.planning.manipulation import ManipulationWorkflow, TaskCancelled
from curobo_ros_python.planning.payload import Payload, pose_matrix, outside_box, configure_payload_links


class Backend:
    def __init__(self,fail=None): self.events=[];self.held=False;self.fail=fail
    def event(self,name):
        self.events.append(name)
        if self.fail==name: raise TaskCancelled(name)
    def validate_pick(self,t): self.event('validate_pick')
    def validate_place(self,t): self.event('validate_place')
    def validate_pair(self,g,p): self.event('validate_pair')
    def gripper(self,t,p,stage): self.event(stage)
    def move(self,t,stage,contact): self.event((stage,contact))
    def attach(self,t): self.held=True;self.event('attach')
    def enable_payload(self,t): self.event('enable_payload')
    def end(self,t): self.event('end')
    def detach(self,t): self.event('detach');self.held=False


TASK=NS(open_position=0.,close_position=-.042)


class WorkflowTest(unittest.TestCase):
    def test_pick_place_reuses_both_stages_and_contact_boundaries(self):
        b=Backend();ManipulationWorkflow(b).pick_place(TASK,TASK)
        self.assertEqual(b.events,['validate_pair','validate_pick','open',('prepick',False),
            ('pick',True),'close','attach',('lift',True),'enable_payload','end','validate_place',
            ('preplace',False),('place',True),'release','detach',('retreat',True),'end'])
        self.assertFalse(b.held)

    def test_close_failure_stops_before_attach(self):
        b=Backend('close')
        with self.assertRaises(TaskCancelled): ManipulationWorkflow(b).pick(TASK)
        self.assertFalse(b.held);self.assertNotIn(('lift',True),b.events)

    def test_release_failure_never_detaches_or_retreats(self):
        b=Backend('release');b.held=True
        with self.assertRaises(TaskCancelled): ManipulationWorkflow(b).place(TASK)
        self.assertTrue(b.held);self.assertNotIn('detach',b.events)

    def test_initial_validation_failure_has_no_commands(self):
        b=Backend('validate_pair')
        with self.assertRaises(TaskCancelled): ManipulationWorkflow(b).pick_place(TASK,TASK)
        self.assertEqual(b.events,['validate_pair'])


class GeometryTest(unittest.TestCase):
    def test_box_spheres_cover_corners_and_follow_tool_offset(self):
        offset=pose_matrix([.1,.2,.3],[1,0,0,0])
        payload=Payload('right_arm',np.array([.1,.2,.3],dtype=np.float32),offset,np.eye(4))
        spheres=payload.spheres()
        for x in (-.05,.05):
            for y in (-.1,.1):
                for z in (-.15,.15):
                    point=np.array([x,y,z],dtype=np.float32)+offset[:3,3]
                    self.assertTrue(np.any(np.linalg.norm(spheres[:,:3]-point,axis=1)<=spheres[:,3]+1e-6))
        self.assertEqual(payload.tool,'right_arm_ee_link')

    def test_obb_cloud_filter_keeps_background(self):
        matrix=pose_matrix([1,0,0],[1,0,0,0])
        np.testing.assert_array_equal(outside_box(np.array([[1,0,0],[0,0,0]],dtype=np.float32),matrix,[.1,.1,.1]),[False,True])
        with self.assertRaises(ValueError): pose_matrix([0,0,0],[0,0,0,0])

    def test_payload_links_do_not_change_original_contact_policy(self):
        robot={'robot_cfg':{'kinematics':dict(tool_frames=['left_arm_ee_link','right_arm_ee_link'],
            collision_link_names=['arm','left_gripper_base','right_gripper_base'],self_collision_ignore={'arm':['other']})}}
        configure_payload_links(robot)
        kin=robot['robot_cfg']['kinematics']
        self.assertEqual(kin['self_collision_ignore']['arm'],['other'])
        self.assertEqual(kin['extra_collision_spheres']['pick_place_right_arm_payload'],27)
        self.assertEqual(kin['extra_links']['pick_place_right_arm_payload']['parent_link_name'],'right_arm_ee_link')


if __name__=='__main__': unittest.main()
