"""Shared pick/place state machine; independent of ROS and GPU libraries."""
from __future__ import annotations


class TaskCancelled(RuntimeError):
    pass


class ManipulationWorkflow:
    """Backend owns geometry, execution and prediction; this class owns ordering."""

    def __init__(self, backend):
        self.backend=backend

    def pick(self, task):
        b=self.backend
        b.validate_pick(task)
        b.gripper(task, task.open_position, 'open')
        b.move(task, 'prepick', contact=False)
        b.move(task, 'pick', contact=True)
        b.gripper(task, task.close_position, 'close')
        b.attach(task)
        b.move(task, 'lift', contact=True)
        b.enable_payload(task)
        b.end(task)

    def place(self, task):
        b=self.backend
        b.validate_place(task)
        b.move(task, 'preplace', contact=False)
        b.move(task, 'place', contact=True)
        b.gripper(task, task.open_position, 'release')
        # Never detach before the opening trajectory has been generated.
        b.detach(task)
        b.move(task, 'retreat', contact=True)
        b.end(task)

    def pick_place(self, pick, place):
        self.backend.validate_pair(pick, place)
        self.pick(pick)
        self.place(place)
