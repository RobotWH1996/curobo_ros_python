"""Focused scene lifecycle checks: no robot I/O or full planner startup."""
import time
import unittest
import xml.etree.ElementTree as ET
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch
from curobo._src.geom.types import SceneCfg, Cuboid
from curobo_ros_python.planning.pick_place_server import TaskBackend


def box(name, x=0., size=.1):
    return Cuboid(name=name,pose=[x,0.,0.,1.,0.,0.,0.],dims=[size]*3)


class PlacedSceneTest(unittest.TestCase):
    def backend(self):
        backend=TaskBackend.__new__(TaskBackend)
        scene=SceneCfg(cuboid=[box('table')])
        motion=NS(scene_collision_checker=NS(scene_model=scene))
        def update(value): motion.scene_collision_checker.scene_model=value
        motion.update_world=Mock(side_effect=update)
        checker=NS(prepare_scene=Mock(return_value={'prepared':object()}),environment={})
        backend.p=NS(_planners={'left_arm':motion},_mesh_scenes={'left_arm':{}},_mesh_checker=checker)
        backend.server=NS(placed={})
        return backend,motion,checker

    def test_unchanged_scene_skips_rebuild_but_add_move_resize_remove_update(self):
        backend,motion,checker=self.backend()
        backend._set_placed()
        motion.update_world.assert_not_called()
        checker.prepare_scene.assert_not_called()
        part=box('placed_part')
        backend.server.placed={'part':part}
        backend._set_placed()
        self.assertEqual(motion.update_world.call_count,1)
        self.assertEqual([b.name for b in checker.prepare_scene.call_args.args[0].objects],['table','placed_part'])
        # A separately copied, geometrically identical object is still unchanged.
        backend.server.placed={'part':box('placed_part')}
        backend._set_placed()
        self.assertEqual(motion.update_world.call_count,1)
        backend.server.placed={'part':box('placed_part',x=.2)}
        backend._set_placed()
        self.assertEqual(motion.update_world.call_count,2)
        backend.server.placed={'part':box('placed_part',x=.2,size=.2)}
        backend._set_placed()
        self.assertEqual(motion.update_world.call_count,3)
        backend.server.placed={}
        backend._set_placed()
        self.assertEqual(motion.update_world.call_count,4)
        self.assertEqual(checker.prepare_scene.call_count,4)
        self.assertEqual([b.name for b in motion.scene_collision_checker.scene_model.cuboid],['table'])
        self.assertEqual([b.name for b in checker.prepare_scene.call_args.args[0].objects],['table'])
        self.assertIs(backend.p._mesh_scenes['left_arm'],checker.prepare_scene.return_value)

    def test_stale_objects_list_is_repaired_before_fcl_preparation(self):
        backend,motion,checker=self.backend()
        backend.server.placed={'part':box('placed_part')}
        motion.scene_collision_checker.scene_model.cuboid.append(box('placed_part'))
        # Mimic the old direct assignment: cuboid and objects disagree.
        self.assertEqual(len(motion.scene_collision_checker.scene_model.objects),1)
        backend._set_placed()
        motion.update_world.assert_called_once()
        self.assertEqual([b.name for b in checker.prepare_scene.call_args.args[0].objects],['table','placed_part'])

    def test_fcl_preparation_failure_keeps_current_world(self):
        backend,motion,checker=self.backend()
        original=motion.scene_collision_checker.scene_model
        backend.server.placed={'part':box('placed_part')}
        checker.prepare_scene.side_effect=RuntimeError('bad scene')
        with self.assertRaisesRegex(RuntimeError,'bad scene'): backend._set_placed()
        motion.update_world.assert_not_called()
        self.assertIs(motion.scene_collision_checker.scene_model,original)
        self.assertEqual(backend.p._mesh_scenes['left_arm'],{})

    def test_preview_cleanup_restores_saved_scenes_without_intermediate_build(self):
        backend,motion,checker=self.backend()
        saved_scene=motion.scene_collision_checker.scene_model
        backend.live=False
        backend.node=NS(_joint_map={'q':0.})
        backend.server.invalidate_world=Mock()
        backend.saved_held=None
        backend.saved_placed={}
        backend.saved_preview=(False,[])
        backend.saved_worlds={'left_arm':saved_scene}
        backend.saved_mesh_scenes={'left_arm':{'original':object()}}
        backend.saved_fcl_environment={'original':object()}
        backend.p.update_gripper_state=Mock()
        backend.server.placed={'part':box('placed_part')}
        with patch('curobo_ros_python.planning.pick_place_server.set_payload'):
            backend.cleanup()
        checker.prepare_scene.assert_not_called()
        motion.update_world.assert_called_once_with(saved_scene)
        self.assertIs(backend.p._mesh_scenes,backend.saved_mesh_scenes)
        self.assertIs(checker.environment,backend.saved_fcl_environment)
        self.assertEqual(backend.p._placed_objects,{})

    def test_feedback_checks_all_observed_joints_and_live_freshness(self):
        backend,_,checker=self.backend()
        checker.ordered=[ET.fromstring(f'<joint name="{name}" type="revolute"/>')
                         for name in ('active_arm','other_arm','finger')]
        backend.node=NS(_joint_map={name:0. for name in ('active_arm','other_arm','finger')},
                        _joint_received={name:time.monotonic() for name in ('active_arm','other_arm','finger')},
                        _p=lambda name:1.)
        backend.live=True
        backend._validate_joint_feedback()
        backend.node._joint_received['other_arm']=0.
        with self.assertRaisesRegex(RuntimeError,'other_arm'): backend._validate_joint_feedback()
        backend.live=False
        backend._validate_joint_feedback()  # Previews allow saved feedback.
        backend.node._joint_map['finger']=float('nan')
        with self.assertRaisesRegex(RuntimeError,'finger'): backend._validate_joint_feedback()


if __name__=='__main__': unittest.main()
