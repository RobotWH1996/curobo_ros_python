"""CPU integration regressions using production methods without ROS startup."""
import ast
import __future__
from pathlib import Path
from threading import Lock
from types import SimpleNamespace as NS
from unittest.mock import Mock
import unittest
import numpy as np
from curobo_ros_python.planning.self_filter import SelfFilterMixin

ROOT = Path(__file__).parents[1] / 'curobo_ros_python'


def methods(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [n for n in ast.walk(tree)
             if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == len(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec',
                 flags=__future__.annotations.compiler_flag), namespace)
    return namespace


PC = methods(ROOT / 'planning/pointcloud.py',
             ['_filter_and_downsample', 'crop_bounds', 'cloud_to_base_points', 'drop_boxes'],
             {'np': np, '_xyz_from_cloud': lambda msg: msg.points,
              'transform_points': lambda pts, translation, quat: pts + np.array(translation)})
cloud_to_base_points = PC['cloud_to_base_points']


class IntegrationTest(unittest.TestCase):
    def fixture(self, mode='apply'):
        node = SelfFilterMixin()
        node.self_filter_mode = mode
        node.self_filter_padding = .01
        node.self_filter_max_processing_ms = 100.
        node._joint_map = {'q': 0.}
        node._self_filter_cache = None
        node._self_filter = NS(required=['q'],
            mask=Mock(side_effect=lambda pts, *args, **kwargs: pts[:, 0] < .2))
        node._self_filters = dict.fromkeys(('left_arm', 'right_arm', 'both_arms'), node._self_filter)
        node._publish_self_filter = Mock()
        node.get_logger = lambda: NS(info=Mock(), warn=Mock())
        node.use_pointcloud = True
        node.pointcloud_topic = '/points'
        node._pc_lock = Lock()
        node._pc_seq = 7
        node._pc_msg = NS(header=NS(frame_id='camera'),
                          points=np.array([[.09, 0., 0.], [.11, 0., 0.], [.31, 0., 0.]], dtype=np.float32))
        node._lookup_cloud_tf = Mock(return_value=([.02, 0., 0.], [1., 0., 0., 0.]))
        node.pointcloud_min_range = -1.
        node.pointcloud_max_range = 10.
        node.pointcloud_pitch = .1
        node.pointcloud_max_points = 100
        node.filter_pointcloud = True
        node.pointcloud_bounds_min = None
        node.pointcloud_bounds_max = None
        node.static_exclude_boxes = lambda: None
        return node

    def cloud(self, node, callback=None):
        return cloud_to_base_points(node._pc_msg, [.02, 0., 0.], [1., 0., 0., 0.],
            -1., 10., .1, node.pointcloud_max_points, filter_cloud=node.filter_pointcloud,
            bounds_min=node.pointcloud_bounds_min, bounds_max=node.pointcloud_bounds_max,
            point_filter=callback)[0]

    def test_observe_and_failure_match_off_with_sampling_and_crop(self):
        for mode in ['observe', 'apply']:
            for capped in [False, True]:
                node = self.fixture(mode)
                if capped:
                    node.pointcloud_max_points = 2
                    node.pointcloud_bounds_min = [.12, -1., -1.]
                    node.pointcloud_bounds_max = [1., 1., 1.]
                if mode == 'apply':
                    node._self_filter.mask.side_effect = TimeoutError('budget')
                expected = self.cloud(node)
                actual = self.cloud(node, lambda pts: node._filter_robot_points(node._pc_msg, pts))
                np.testing.assert_array_equal(actual, expected)
                node._self_filter.mask.assert_called_once()

    def test_apply_still_removes_points(self):
        node = self.fixture()
        actual = self.cloud(node, lambda pts: node._filter_robot_points(node._pc_msg, pts))
        self.assertEqual(len(actual), 1)
        self.assertGreater(actual[0, 0], .2)
        self.assertTrue(node._self_filter_succeeded)

    def follower(self, node):
        class TransformError(Exception):
            pass
        ns = methods(ROOT / 'planning/mpc_follow.py', ['_sync_world', '_clear_loaded_world'],
                     {'np': np, 'TransformException': TransformError,
                      'cloud_to_base_points': cloud_to_base_points, 'drop_boxes': PC['drop_boxes'],
                      '_cloud_grid_note': lambda *args: '', 'publish_collision_world': Mock()})
        cls = type('Follower', (), {name: ns[name] for name in ['_sync_world', '_clear_loaded_world']})
        follower = cls()
        follower._node = node
        follower._loaded_pc_seq = None
        follower._filter_retry_seq = None
        return follower, TransformError

    def test_failed_same_frame_retries_then_stops_after_success(self):
        for fault in ['missing', 'timeout']:
            node = self.fixture()
            if fault == 'missing':
                node._joint_map = {}
            else:
                node._self_filter.mask.side_effect = TimeoutError('budget')
            follower, _ = self.follower(node)
            mpc = NS(update_world_from_points=Mock(), clear_world=Mock())
            self.assertIsNone(follower._sync_world(mpc, publish=False))
            self.assertEqual(follower._loaded_pc_seq, 7)
            self.assertEqual(follower._filter_retry_seq, 7)
            follower._sync_world(mpc, publish=False)
            self.assertEqual(mpc.update_world_from_points.call_count, 1)
            node._joint_map = {'q': 0.}
            node._self_filter.mask.side_effect = lambda pts, *a, **k: pts[:, 0] < .2
            follower._sync_world(mpc, publish=False)
            self.assertIsNone(follower._filter_retry_seq)
            self.assertEqual(mpc.update_world_from_points.call_count, 2)
            self.assertEqual(len(mpc.update_world_from_points.call_args.args[0]), 1)
            follower._sync_world(mpc, publish=False)
            self.assertEqual(mpc.update_world_from_points.call_count, 2)

    def test_tf_failure_retries_and_all_removed_clears_world(self):
        node = self.fixture()
        follower, error = self.follower(node)
        mpc = NS(update_world_from_points=Mock(), clear_world=Mock())
        node._lookup_cloud_tf.side_effect = error('missing')
        self.assertIn('TF', follower._sync_world(mpc, publish=False))
        self.assertIsNone(follower._loaded_pc_seq)
        node._lookup_cloud_tf.side_effect = None
        node._self_filter.mask.side_effect = lambda pts, *a, **k: np.ones(len(pts), dtype=bool)
        self.assertIsNone(follower._sync_world(mpc, publish=False))
        mpc.clear_world.assert_called_once()
        self.assertIsNone(follower._filter_retry_seq)
        follower._sync_world(mpc, publish=False)
        mpc.clear_world.assert_called_once()


if __name__ == '__main__':
    unittest.main()
