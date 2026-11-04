"""Arm-only masking and same-frame planning-group switches."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch
import numpy as np
from curobo_ros_python.planning.groups import GROUP_JOINTS
from curobo_ros_python.planning.mesh_collision import MeshCollisionChecker
from curobo_ros_python.planning.self_filter import MeshSelfFilter
import test_self_filter_integration as integration


class ArmGeometryTest(unittest.TestCase):
    def test_selected_arm_includes_tool_excludes_torso_and_other_arm(self):
        links = ''.join(f'<link name="{name}"><collision><geometry><box size=".2 .2 .2"/></geometry></collision></link>'
                        for name in ('base','torso','left','right','tool'))
        def joint(name, parent, child, xyz):
            return f'<joint name="{name}" type="prismatic"><parent link="{parent}"/><child link="{child}"/><origin xyz="{xyz}"/><axis xyz="1 0 0"/></joint>'
        joints = (joint('torso_q','base','torso','0 0 0')
                  + joint('left_q','torso','left','-2 0 0')
                  + joint('right_q','torso','right','2 0 0')
                  + joint('finger_q','left','tool','0 0 1'))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'robot.urdf'
            path.write_text('<robot name="test">'+links+joints+'</robot>')
            config=dict(base_link='base',collision_link_names=['torso','left','right','tool'],
                        cspace=dict(joint_names=['torso_q','left_q','right_q','finger_q'],default_joint_position=[0.]*4))
            checker=MeshCollisionChecker(str(path),config,retain_meshes=True)
            original_names=list(checker.names)
            original_order=list(checker.ordered)
            points=np.array([[-2,0,0],[2,0,0],[0,0,0],[-2,0,1]],dtype=np.float32)
            expected={'left_arm':[1,0,0,1],'right_arm':[0,1,0,0],'both_arms':[1,1,0,1]}
            tables={'left_arm':['left_q'],'right_arm':['right_q'],'both_arms':['left_q','right_q']}
            cache={};filters={}
            with patch.dict(GROUP_JOINTS,tables,clear=True):
                for group,mask in expected.items():
                    filter_=MeshSelfFilter(checker,group,vertex_tree_cache=cache)
                    filters[group]=filter_
                    positions={name:0. for name in filter_.required}
                    np.testing.assert_array_equal(filter_.mask(points,positions,.01),mask)
                    self.assertNotIn('torso',filter_.checker.names)
                    if group=='left_arm': self.assertNotIn('right_q',filter_.required)
                    if group=='right_arm': self.assertNotIn('left_q',filter_.required)
                with self.assertRaises(ValueError): MeshSelfFilter(checker,'invalid')
            both=filters['both_arms']
            for group in ('left_arm','right_arm'):
                arm=filters[group]
                for index,name in enumerate(arm.checker.names):
                    shared_index=both.checker.names.index(name)
                    self.assertIs(arm.vertex_trees[index],both.vertex_trees[shared_index])
                    self.assertIsNot(arm.local_objects[index],both.local_objects[shared_index])
            self.assertEqual(len(cache),3)
            self.assertEqual(checker.names,original_names)
            self.assertEqual(checker.ordered,original_order)

    def test_group_changes_invalidate_mask_cache(self):
        helper=integration.IntegrationTest()
        node=helper.fixture()
        left=NS(required=['q'],mask=Mock(return_value=np.array([True,False])))
        right=NS(required=['q'],mask=Mock(return_value=np.array([False,True])))
        node._self_filters={'left_arm':left,'right_arm':right}
        pts=np.array([[0,0,0],[1,0,0]],dtype=np.float32)
        np.testing.assert_array_equal(node._filter_robot_points(node._pc_msg,pts,'left_arm'),pts[1:])
        np.testing.assert_array_equal(node._filter_robot_points(node._pc_msg,pts,'right_arm'),pts[:1])
        left.mask.assert_called_once();right.mask.assert_called_once()

    def test_mpc_group_switch_reloads_same_cloud_and_target_selects_arm(self):
        helper=integration.IntegrationTest()
        node=helper.fixture()
        follower,_=helper.follower(node)
        mpc=NS(update_world_from_points=Mock(),clear_world=Mock())
        original=node._filter_robot_points
        node._filter_robot_points=Mock(wraps=original)
        for group in ('left_arm','right_arm','both_arms'):
            follower._sync_world(mpc,publish=False,group_name=group)
            self.assertEqual(node._filter_robot_points.call_args.args[2],group)
        self.assertEqual(mpc.update_world_from_points.call_count,3)
        follower._target=NS(has_left=True,has_right=False)
        follower._sync_world(mpc,publish=False)
        self.assertEqual(node._filter_robot_points.call_args.args[2],'left_arm')
        follower._target=NS(has_left=False,has_right=True)
        follower._sync_world(mpc,publish=False)
        self.assertEqual(node._filter_robot_points.call_args.args[2],'right_arm')


if __name__=='__main__': unittest.main()
