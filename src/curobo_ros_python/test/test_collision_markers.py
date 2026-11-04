"""障碍体素：环境格子保持橙色，本体滤除格子改成蓝色。"""
import unittest

import numpy as np
from visualization_msgs.msg import Marker

from curobo_ros_python.visualization.rviz import collision_voxels_to_markers


class CollisionMarkerColorTests(unittest.TestCase):
    def markers(self, environment, body):
        array = collision_voxels_to_markers(
            np.asarray(environment, dtype=np.float64),
            pitch=0.05,
            body_points=np.asarray(body, dtype=np.float64),
        )
        return {marker.id: marker for marker in array.markers if marker.action == Marker.ADD}

    def test_environment_stays_orange_and_body_is_blue(self):
        drawn = self.markers([[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]])
        self.assertGreater(drawn[1].color.r, drawn[1].color.b)
        self.assertGreater(drawn[2].color.b, drawn[2].color.r)
        self.assertEqual(len(drawn[1].points), 1)
        self.assertEqual(len(drawn[2].points), 1)

    def test_shared_cell_is_shown_as_filtered_body(self):
        drawn = self.markers([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [[0.01, 0.0, 0.0]])
        self.assertEqual(len(drawn[1].points), 1)
        self.assertEqual(len(drawn[2].points), 1)
        self.assertGreater(drawn[2].color.b, 0.5)

    def test_empty_clouds_only_clear_previous_markers(self):
        array = collision_voxels_to_markers(np.zeros((0, 3)), pitch=0.05, body_points=np.zeros((0, 3)))
        self.assertEqual(len(array.markers), 1)
        self.assertEqual(array.markers[0].action, Marker.DELETEALL)


if __name__ == "__main__":
    unittest.main()
