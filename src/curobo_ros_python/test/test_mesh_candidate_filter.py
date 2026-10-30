"""Real FCL geometry and IK seed filtering regression tests (CPU only)."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np
import torch
import trimesh

from curobo_ros_python.planning.mesh_collision import MeshCollisionChecker
from curobo_ros_python.planning.ik_collision_filter import install_ik_filter


class GeometryTest(unittest.TestCase):
    def test_lycus_fcl_policy_is_independent_of_gpu_policy(self):
        import copy
        from curobo.config_io import load_yaml
        from curobo.content import get_robot_configs_path
        source = load_yaml(str(Path(get_robot_configs_path()) / 'lycus.yml'))
        original = copy.deepcopy(source)
        with patch('curobo.config_io.load_yaml', return_value=source), \
             patch.object(MeshCollisionChecker, '__init__', return_value=None) as constructor:
            MeshCollisionChecker.from_robot('lycus.yml')
        mesh_config = constructor.call_args.args[1]
        for side in ('L', 'R'):
            a = f'AR5-5_08{side}-W4C1C5-ZY2_link5'
            b = f'AR5-5_08{side}-W4C1C5-ZY2_link7'
            pairs = {frozenset((x, y)) for x, names in mesh_config['self_collision_ignore'].items()
                     for y in names}
            self.assertNotIn(frozenset((a, b)), pairs)
        self.assertEqual(source, original)

    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        path = Path(self.directory.name) / 'robot.urdf'
        path.write_text('''<robot name="fixture">
          <link name="base"><collision><geometry><box size=".4 .4 .4"/></geometry></collision></link>
          <link name="tool"><collision><geometry><box size=".4 .4 .4"/></geometry></collision></link>
          <joint name="slide" type="prismatic"><parent link="base"/><child link="tool"/>
            <origin xyz="1 0 0"/><axis xyz="1 0 0"/></joint>
          <link name="finger"><collision><geometry><box size=".1 .1 .1"/></geometry></collision></link>
          <joint name="jaw" type="prismatic"><parent link="base"/><child link="finger"/>
            <origin xyz="2 0 0"/><axis xyz="1 0 0"/><mimic joint="slide" multiplier="-1"/></joint>
        </robot>''')
        self.config = dict(base_link='base', collision_link_names=['base', 'tool', 'finger'],
                           cspace=dict(joint_names=['slide'], default_joint_position=[0]),
                           self_collision_ignore={})
        self.checker = MeshCollisionChecker(str(path), self.config)

    def test_self_collision_clear_then_intersect_then_clear(self):
        self.assertIsNone(self.checker.check({'slide': 0}))
        hit = self.checker.check({'slide': -.9})
        self.assertEqual((hit.kind, hit.first, hit.second), ('self', 'base', 'tool'))
        self.assertIsNone(self.checker.check({'slide': .2}))

    def test_environment_replacement_uses_obstacle_pose(self):
        def obstacle(center):
            def mesh(**kwargs):
                result = trimesh.creation.box([.4, .4, .4])
                result.apply_translation(center)
                return result
            return NS(name='wall', get_trimesh_mesh=mesh)
        self.checker.environment = self.checker.prepare_scene(NS(objects=[obstacle([1.2, 0, 0])]))
        self.assertEqual(self.checker.check({'slide': 0}).kind, 'environment')
        self.checker.environment = self.checker.prepare_scene(NS(objects=[obstacle([3, 0, 0])]))
        self.assertIsNone(self.checker.check({'slide': 0}))
        self.checker.environment = {}
        self.assertIsNone(self.checker.check({'slide': 0}))

    def test_invalid_state_and_unsupported_scene_fail(self):
        with self.assertRaises(ValueError):
            self.checker.check({'slide': float('nan')})
        with self.assertRaises(ValueError):
            self.checker.prepare_scene(NS(voxel=[object()]))

    def test_pair_exclusion_and_mimic(self):
        self.checker.pairs = np.empty((0, 2), dtype=int)
        self.assertIsNone(self.checker.check({'slide': -.9}))
        # Mimic input must not override the driven joint, even if in feedback.
        from curobo_ros_python.planning.mesh_collision import _origin
        self.assertEqual(_origin(None).shape, (4, 4))
        tf = self.checker._transforms({'slide': .3, 'jaw': 100})
        self.assertAlmostEqual(tf[-1, 0, 3], 1.7)

    def test_full_body_checker_obeys_policy_without_link_specific_override(self):
        prefix = 'AR5-5_08L-W4C1C5-ZY2_'
        link5, link7 = prefix + 'link5', prefix + 'link7'
        path = Path(self.directory.name) / 'required_pair.urdf'
        path.write_text(f'''<robot name="required-pair">
          <link name="base"/>
          <link name="{link5}"><collision><geometry><box size=".1 .1 .1"/></geometry></collision></link>
          <joint name="to5" type="fixed"><parent link="base"/><child link="{link5}"/></joint>
          <link name="{link7}"><collision><geometry><box size=".1 .1 .1"/></geometry></collision></link>
          <joint name="to7" type="fixed"><parent link="{link5}"/><child link="{link7}"/></joint>
        </robot>''')
        checker = MeshCollisionChecker(str(path), dict(
            base_link='base', collision_link_names=[link5, link7],
            cspace=dict(joint_names=[], default_joint_position=[]),
            self_collision_ignore={link5: [link7]}))
        pairs = {(checker.names[i], checker.names[j]) for i, j in checker.pairs}
        self.assertNotIn((link5, link7), pairs)
        checker = MeshCollisionChecker(str(path), dict(
            base_link='base', collision_link_names=[link5, link7],
            cspace=dict(joint_names=[], default_joint_position=[]),
            self_collision_ignore={}))
        self.assertEqual(checker.check({}).kind, 'self')


class State:
    def __init__(self, position):
        self.position = position
        self.joint_names = ['locked', 'slide']

    def clone(self):
        return State(self.position.clone())


class SeedFilterTest(unittest.TestCase):
    def run_filter(self, q, good, check):
        solution = torch.tensor(q, dtype=torch.float32).reshape(1, -1, 1)
        success = torch.tensor([good], dtype=torch.bool)
        full = torch.cat([torch.full_like(solution, 7), solution], dim=-1)
        result = NS(solution=solution, success=success, feasible=success.clone(),
                    js_solution=State(full), batch_size=1)
        solver = NS(joint_names=['slide'], solve_pose=lambda *a, **kw: result)
        reports = []
        install_ik_filter(solver, check, lambda n, rejected: reports.append((n, len(rejected))))
        return solver.solve_pose(), solution, success, reports

    def test_rejected_candidates_never_remain_as_optimizer_seeds(self):
        result, original, success, reports = self.run_filter(
            [1, 2, 3], [True, True, False],
            lambda q: 'self collision' if q['slide'] == 1 else None)
        self.assertEqual(result.success.tolist(), [[False, True, False]])
        self.assertEqual(result.solution.flatten().tolist(), [2, 2, 2])
        self.assertEqual(result.js_solution.position[0, :, 0].tolist(), [7, 7, 7])
        self.assertEqual(result.js_solution.position[0, :, 1].tolist(), [2, 2, 2])
        self.assertEqual(original.flatten().tolist(), [1, 2, 3])
        self.assertEqual(success.tolist(), [[True, True, False]])
        self.assertEqual(reports, [(2, 1)])

    def test_all_rejected_returns_no_success(self):
        result, _, _, reports = self.run_filter([1, 2], [True, True], lambda q: 'environment collision')
        self.assertFalse(result.success.any())
        self.assertEqual(reports, [(2, 2)])

    def test_passed_candidates_preserved(self):
        result, original, _, reports = self.run_filter([1, 2], [True, True], lambda q: None)
        torch.testing.assert_close(result.solution, original)
        self.assertEqual(reports, [(2, 0)])


if __name__ == '__main__':
    unittest.main()
