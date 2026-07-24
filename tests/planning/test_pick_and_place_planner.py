from api import RobotController
from planning.models import MotionCommand, PlanningFailure, TargetPose
from planning.pick_and_place_planner import PickAndPlacePlanner
from tests.policy_helpers import PERMISSIVE_SINGULARITY_POLICY


class RecordingSink:
    def __init__(self) -> None:
        self.commands: list[MotionCommand] = []

    def send(self, command: MotionCommand) -> None:
        self.commands.append(command)


class HardwareMarkedRecordingSink(RecordingSink):
    requires_hardware_safe_prevalidation = True


def test_invalid_future_deposit_sends_zero_commands() -> None:
    sink = RecordingSink()
    planner = PickAndPlacePlanner(
        enforce_hardware_safe_limits=False,
        singularity_policy=PERMISSIVE_SINGULARITY_POLICY,
    )
    planner.poses["cartesian_targets"]["drop_off"]["x_mm"] = 10_000.0
    controller = RobotController(sink, planner=planner)

    success = controller.run_pick_and_place(230.0, 180.0, 60.0)

    assert not success
    assert sink.commands == []
    assert controller.machine is None
    assert isinstance(controller.last_planning_failure, PlanningFailure)
    assert controller.last_planning_failure.waypoint == "deposit"
    assert controller.last_planning_failure.code == "WORKSPACE_VIOLATION"


def test_state_machine_executes_exact_commands_from_accepted_plan() -> None:
    sink = RecordingSink()
    planner = PickAndPlacePlanner(
        enforce_hardware_safe_limits=False,
        singularity_policy=PERMISSIVE_SINGULARITY_POLICY,
    )
    expected = planner.plan(TargetPose(230.0, 180.0, 60.0))
    assert not isinstance(expected, PlanningFailure)
    controller = RobotController(sink, planner=planner)

    assert controller.run_pick_and_place(230.0, 180.0, 60.0)

    assert sink.commands == list(expected.commands)


def test_hardware_sink_marker_enables_strict_prevalidation() -> None:
    sink = HardwareMarkedRecordingSink()
    controller = RobotController(
        sink,
        planner=PickAndPlacePlanner(
            singularity_policy=PERMISSIVE_SINGULARITY_POLICY,
        ),
    )

    success = controller.run_pick_and_place(230.0, 180.0, 60.0)

    assert not success
    assert sink.commands == []
    assert (
        controller.last_planning_failure.code
        == "PHYSICAL_CALIBRATION_REQUIRED"
    )
