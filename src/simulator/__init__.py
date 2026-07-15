"""Adapters shared by robot simulators."""

from simulator.coordinate_frames import robot_to_webots, webots_to_robot
from simulator.webots_motion_sink import WebotsMotionSink, WebotsSimulationEnded

__all__ = [
    "WebotsMotionSink",
    "WebotsSimulationEnded",
    "robot_to_webots",
    "webots_to_robot",
]
