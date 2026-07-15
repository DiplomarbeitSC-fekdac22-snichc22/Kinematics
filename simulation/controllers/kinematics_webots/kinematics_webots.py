#!/usr/bin/env python3
"""Run the repository pick-and-place state machine inside Webots."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# simulation/controllers/kinematics_webots -> repository root
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from controller import Supervisor

from api import RobotController
from simulator.webots_motion_sink import (
    WebotsMotionSink,
    WebotsSimulationEnded,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one Kinematics pick-and-place sequence in Webots."
    )
    parser.add_argument(
        "--target-mm",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=(230.0, 180.0, 60.0),
        help="Target in the project robot frame (millimetres).",
    )
    parser.add_argument(
        "--quit-when-done",
        action="store_true",
        help="Ask the Supervisor API to close the simulation after the run.",
    )
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    robot = Supervisor()

    try:
        sink = WebotsMotionSink(robot)
        controller = RobotController(sink)
        x_mm, y_mm, z_mm = args.target_mm

        print(
            "[WEBOTS] Starting repository state machine at target "
            f"x={x_mm:.1f}, y={y_mm:.1f}, z={z_mm:.1f} mm"
        )
        success = controller.run_pick_and_place(x_mm, y_mm, z_mm)
        print(
            "[WEBOTS] Sensor snapshot: "
            + json.dumps(sink.sensor_snapshot(), sort_keys=True)
        )
        print(
            "WEBOTS_SMOKE_TEST: PASS"
            if success
            else "WEBOTS_SMOKE_TEST: FAIL"
        )
    except (TimeoutError, ValueError, KeyError, WebotsSimulationEnded) as exc:
        print(f"WEBOTS_SMOKE_TEST: FAIL - {exc}", file=sys.stderr)
        success = False

    quit_requested = args.quit_when_done or os.environ.get(
        "KINEMATICS_WEBOTS_QUIT_WHEN_DONE"
    ) == "1"

    if quit_requested:
        robot.simulationQuit(0 if success else 1)
        return 0 if success else 1

    while robot.step(int(robot.getBasicTimeStep())) != -1:
        pass

    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
