"""Run with the cuRobo Python after sourcing ROS and this workspace."""
import copy
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch
import numpy as np
import torch
import yaml
from rosidl_runtime_py.set_message import set_message_fields
from rclpy.action import GoalResponse
from curobo_ros_interfaces.action import Pick, Place, PickPlace
from curobo_ros_python.motion_planning_node import CuroboMotionPlanningNode, _PARAM_DEFAULTS
from curobo_ros_python.planning.groups import apply_robot
from curobo_ros_python.planning.manipulation import ManipulationWorkflow
from curobo_ros_python.planning.payload import Payload
from curobo_ros_python.planning.pick_collision import write_payload_spheres
from curobo_ros_python.planning.pick_place_server import PickPlaceServer
from curobo_ros_python.planning.robot_profile import active_profile
from curobo_ros_python.planning.task_geometry import TaskGeometry
from curobo_ros_python.planning.trajectory import TrajectoryExecutor

ROOT = Path(__file__).resolve().parents[1]


class RobotConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(apply_robot, active_profile().robot_id)

    def test_executor_reads_selected_robot_after_import(self):
        node = Mock()
        with patch('curobo_ros_python.planning.trajectory.ActionClient'):
            apply_robot('k100.yml')
            self.assertEqual(set(TrajectoryExecutor(node)._actions),
                             {'left_arm', 'right_arm', 'both_arms'})
            apply_robot('lycus.yml')
            actions = TrajectoryExecutor(node)._actions
            self.assertEqual(actions['left_gripper'],
                             '/left_gripper_controller/follow_joint_trajectory')
            self.assertEqual(actions['right_gripper'],
                             '/right_gripper_controller/follow_joint_trajectory')

    def test_k100_rejects_all_grasp_goals_before_preparation(self):
        apply_robot('k100.yml')
        logger = Mock()
        node = NS(_p=lambda _: 1.0, get_logger=lambda: logger)
        opt = PickPlaceServer.__new__(PickPlaceServer)
        opt.node, opt.busy = node, False
        for kind, goal, tasks in (
            ('pick', Pick.Goal(), ('task',)),
            ('place', Place.Goal(), ('task',)),
            ('pick_place', PickPlace.Goal(), ('pick', 'place')),
        ):
            for task in tasks:
                getattr(goal, task).group_name = 'left_arm'
            self.assertEqual(opt.accept(kind, goal), GoalResponse.REJECT)
            self.assertIn('k100', logger.warn.call_args.args[0])
        self.assertFalse(opt.busy)

    def test_cloud_frame_follows_robot_with_explicit_override_and_header_fallback(self):
        for robot, configured, expected in (
            ('lycus.yml', 'auto', 'camera_01_color_optical_frame'),
            ('k100.yml', 'auto', 'message_frame'),
            ('lycus.yml', '', 'message_frame'),
            ('k100.yml', '/calibrated_optical_frame', 'calibrated_optical_frame'),
        ):
            with self.subTest(robot=robot, configured=configured):
                apply_robot(robot)
                params = dict(_PARAM_DEFAULTS, pointcloud_child_frame=configured, use_pointcloud=False)
                node = NS(_p=params.__getitem__, _vec3=params.__getitem__, base_frame='base_link',
                          _setup_self_filter=Mock(), self_filter_mode='off',
                          get_logger=Mock(), _tf_buffer=Mock())
                CuroboMotionPlanningNode._setup_pointcloud(node)
                CuroboMotionPlanningNode._lookup_cloud_tf(node, 'message_frame')
                self.assertEqual(node._tf_buffer.lookup_transform.call_args.args[:2],
                                 ('base_link', expected))

class GeometryAndWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(apply_robot, active_profile().robot_id)
        apply_robot('lycus.yml')

    def test_existing_yaml_joint_end_and_pose_route(self):
        goal = PickPlace.Goal()
        set_message_fields(goal, yaml.safe_load((ROOT/'src/curobo_ros_python/config/pick_place.yaml').read_text()))
        backend = TaskGeometry()
        backend.node = NS(_pose_in_base=copy.deepcopy, _p=lambda _: [0., 0., 0.])
        backend.p = NS(_gripper_state=NS(_limits={'left_gripper': (-.042, 0.)}))
        backend.check = Mock()
        backend._preflight = Mock()
        backend.validate_pick(goal.pick)
        backend._validate_place_fields(goal.place)
        self.assertEqual(backend.pick_end.kind, 'joint')
        self.assertEqual(len(backend.pick_end.positions), 7)
        self.assertIsNone(backend.place_end)
        self.assertTrue(np.all(backend.object_size > 0.))
        goal.pick.end_pose.joint_positions = [float('nan')]*7
        with self.assertRaises(ValueError):
            backend.validate_pick(goal.pick)

    def test_payload_updates_preserve_addresses(self):
        spheres = torch.zeros((1, 54, 4))
        kin = NS(link_spheres=spheres, link_name_to_idx_map={
            'pick_place_left_arm_payload': 0, 'pick_place_right_arm_payload': 1})
        kin.get_sphere_index_from_link_name = lambda name: list(range(0, 27)) if 'left' in name else list(range(27, 54))
        pointer = spheres.data_ptr()
        payload = Payload('left_arm', [.1, .1, .1], np.eye(4), np.eye(4))
        write_payload_spheres([kin], payload)
        self.assertTrue(bool((spheres[0, :27, 3] > 0.).all()))
        self.assertTrue(bool((spheres[0, 27:, 3] < 0.).all()))
        write_payload_spheres([kin], None)
        self.assertEqual(pointer, spheres.data_ptr())
        self.assertTrue(bool((spheres[..., 3] < 0.).all()))

    def test_failed_close_never_attaches_or_lifts(self):
        backend = Mock()
        backend.gripper.side_effect = [None, RuntimeError('gripper failed')]
        task = NS(open_position=0., close_position=-.025, group_name='left_arm')
        with self.assertRaisesRegex(RuntimeError, 'gripper failed'):
            ManipulationWorkflow(backend).pick(task)
        backend.attach.assert_not_called()
        self.assertEqual([call.args[1] for call in backend.move.call_args_list], ['prepick', 'pick'])

    def test_failed_release_keeps_payload(self):
        backend = Mock()
        backend.gripper.side_effect = RuntimeError('release failed')
        with self.assertRaises(RuntimeError):
            ManipulationWorkflow(backend).place(NS(open_position=0., group_name='left_arm'))
        backend.detach.assert_not_called()
        self.assertEqual([call.args[1] for call in backend.move.call_args_list], ['preplace', 'place'])

    def test_complete_stage_order(self):
        backend = Mock()
        task = NS(open_position=0., close_position=-.025, group_name='left_arm')
        ManipulationWorkflow(backend).pick_place(task, task)
        self.assertEqual([call.args[1] for call in backend.move.call_args_list],
                         ['prepick', 'pick', 'lift', 'preplace', 'place', 'retreat'])
        self.assertEqual([call.args[2] for call in backend.gripper.call_args_list], ['open', 'close', 'release'])


if __name__ == '__main__':
    unittest.main()
