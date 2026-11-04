"""Offline self-mask geometry, current-state and fail-safe policy tests."""
import unittest
import copy
import xml.etree.ElementTree as ET
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch
from pathlib import Path
import ast
import numpy as np
import trimesh
from curobo_ros_python.planning.self_filter import MeshSelfFilter, SelfFilterMixin


class GeometryTest(unittest.TestCase):
    def fixture(self, mesh=None, transform=None):
        joint = ET.fromstring('<joint name="q" type="prismatic"/>')
        matrix = np.eye(4) if transform is None else transform
        checker = NS(meshes=[trimesh.creation.box() if mesh is None else mesh], names=['body'],
                     ordered=[joint], _transforms=lambda positions: np.array([matrix]))
        return MeshSelfFilter(checker)

    def test_inside_surface_margin_and_nearby_obstacle(self):
        filter_ = self.fixture()
        points = np.array([[0, 0, 0], [.5, 0, 0], [.505, 0, 0], [.52, 0, 0], [2, 0, 0]])
        np.testing.assert_array_equal(filter_.mask(points, {'q': 0.}, .01), [1, 1, 1, 0, 0])

    def test_actual_transform_and_empty_input(self):
        transform = trimesh.transformations.rotation_matrix(.4, [0, 0, 1])
        transform[:3, 3] = [1, 2, 3]
        filter_ = self.fixture(transform=transform)
        np.testing.assert_array_equal(filter_.mask([[1, 2, 3], [0, 0, 0]], {'q': 0.}, .01), [1, 0])
        self.assertEqual(len(filter_.mask(np.empty((0, 3)), {'q': 0.}, .01)), 0)

    def test_open_mesh_does_not_claim_unknown_interior(self):
        mesh = trimesh.Trimesh(vertices=[[0,0,0],[1,0,0],[0,1,0]], faces=[[0,1,2]], process=False)
        filter_ = self.fixture(mesh=mesh)
        self.assertEqual(filter_.open_links, ['body'])
        np.testing.assert_array_equal(filter_.mask([[.2,.2,.005],[.2,.2,.02]], {'q': 0.}, .01), [1, 0])


class PolicyTest(unittest.TestCase):
    def fixture(self, mode):
        class Node(SelfFilterMixin):
            pass
        node = Node()
        node.self_filter_mode = mode
        node.self_filter_padding = .01
        node.self_filter_max_processing_ms = 100.
        node._joint_map = {'q': 0.}
        node._self_filter = NS(required=['q'], mask=Mock(return_value=np.array([True, False])))
        node._self_filters = dict.fromkeys(('left_arm', 'right_arm', 'both_arms'), node._self_filter)
        node._self_filter_cache = None
        node._publish_self_filter = Mock()
        node.get_logger = lambda: NS(info=Mock(), warn=Mock())
        return node, NS(header=NS(stamp=NS(sec=1, nanosec=0)))

    def test_observe_never_changes_planning_points(self):
        node, msg = self.fixture('observe')
        points = np.array([[0.,0.,0.],[2.,0.,0.]])
        self.assertIs(node._filter_robot_points(msg, points), points)

    def test_apply_and_repeated_frame_cache(self):
        node, msg = self.fixture('apply')
        points = np.array([[0.,0.,0.],[2.,0.,0.]])
        np.testing.assert_equal(node._filter_robot_points(msg, points), points[1:])
        repeated=copy.deepcopy(msg)
        repeated.header.stamp.sec+=1
        node._filter_robot_points(repeated, points.copy())
        node._self_filter.mask.assert_called_once()

    def test_missing_invalid_feedback_or_geometry_keeps_original(self):
        for fault in ['state', 'invalid', 'geometry']:
            node, msg = self.fixture('apply')
            if fault == 'state':
                node._joint_map = {}
            elif fault == 'invalid':
                node._joint_map = {'q': float('nan')}
            else:
                node._self_filter.mask.side_effect = RuntimeError('geometry')
            points = np.array([[0.,0.,0.],[2.,0.,0.]])
            self.assertIs(node._filter_robot_points(msg, points), points)
            self.assertFalse(node._self_filter_all_removed)


    def test_timeout_keeps_frame(self):
        node, msg = self.fixture('apply')
        node._self_filter.mask.side_effect = TimeoutError('budget')
        points = np.zeros((2, 3))
        self.assertIs(node._filter_robot_points(msg, points), points)
        self.assertIsNone(node._self_filter_cache)
        self.assertFalse(node._self_filter_all_removed)

    def test_cache_invalidation(self):
        node, msg = self.fixture('apply')
        points = np.zeros((2, 3))
        node._filter_robot_points(msg, points)
        node.self_filter_padding = .02
        node._filter_robot_points(msg, points)
        points[1, 0] = 3.
        node._filter_robot_points(msg, points)
        self.assertEqual(node._self_filter.mask.call_count, 3)

    def test_cloud_timestamp_does_not_affect_filtering(self):
        for stamp in [0, 1, 1700000000]:
            node, msg = self.fixture('apply')
            msg.header.stamp.sec = stamp
            points = np.array([[0., 0., 0.], [2., 0., 0.]])
            np.testing.assert_equal(node._filter_robot_points(msg, points), points[1:])
            self.assertEqual(node._self_filter.mask.call_args.args[1], {'q': 0.})

    def test_current_feedback_change_recomputes_mask(self):
        node, msg = self.fixture('apply')
        points = np.zeros((2, 3))
        node._filter_robot_points(msg, points)
        node._joint_map['q'] = 1.
        node._filter_robot_points(msg, points)
        self.assertEqual(node._self_filter.mask.call_count, 2)
        self.assertEqual(node._self_filter.mask.call_args.args[1], {'q': 1.})

    def test_all_removed_and_off(self):
        node, msg = self.fixture('apply')
        node._self_filter.mask.return_value = np.array([True, True])
        points = np.zeros((2, 3))
        self.assertEqual(node._filter_robot_points(msg, points).shape, (0, 3))
        self.assertTrue(node._self_filter_all_removed)
        node.self_filter_mode = 'off'
        self.assertIs(node._filter_robot_points(msg, points), points)
        self.assertFalse(node._self_filter_all_removed)


class SetupTest(unittest.TestCase):
    def test_declared_defaults_support_disabled_startup(self):
        path = Path(__file__).parents[1] / 'curobo_ros_python/motion_planning_node.py'
        tree = ast.parse(path.read_text())
        defaults = next(ast.literal_eval(n.value) for n in tree.body
                        if isinstance(n, ast.AnnAssign)
                        and isinstance(n.target, ast.Name) and n.target.id == '_PARAM_DEFAULTS')
        node = SelfFilterMixin()
        node._p = defaults.__getitem__
        node._setup_self_filter()
        self.assertEqual(node.self_filter_mode, 'off')
        self.assertGreater(node.self_filter_max_processing_ms, 0)

    def test_geometry_budget_expiry(self):
        filter_ = GeometryTest().fixture()
        with patch('curobo_ros_python.planning.self_filter.time.perf_counter',
                   side_effect=[0., 1.]):
            with self.assertRaises(TimeoutError):
                filter_.mask([[0., 0., 0.]], {'q': 0.}, .01, budget_ms=1.)


if __name__ == '__main__':
    unittest.main()
