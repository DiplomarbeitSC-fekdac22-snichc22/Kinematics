import pytest

from robot_testing.fault_injection import (
    FaultInjectingMotionSink,
    FaultPhase,
    FaultRule,
    RecordingMotionSink,
)
from planning.models import MotionPlan, PlanningFailure, TargetPose
from planning.pick_and_place_planner import PickAndPlacePlanner
from state_machine.pick_and_place import PickAndPlaceStateMachine


SEQUENCE = [
    "move_ready",
    "move_in_front_of_object",
    "advance_towards_object",
    "close_gripper",
    "lift_object",
    "retract_from_shelf",
    "move_deposit",
    "open_gripper",
    "move_home",
]


def accepted_plan() -> MotionPlan:
    result = PickAndPlacePlanner(
        enforce_hardware_safe_limits=False,
    ).plan(TargetPose(230.0, 180.0, 60.0))
    assert not isinstance(result, PlanningFailure)
    return result


def make_machine(
    failed_command: str,
    *,
    phase: FaultPhase = FaultPhase.BEFORE_SEND,
) -> tuple[PickAndPlaceStateMachine, FaultInjectingMotionSink, RecordingMotionSink]:
    recorder = RecordingMotionSink()
    injecting_sink = FaultInjectingMotionSink(
        recorder,
        [
            FaultRule(
                command_name=failed_command,
                phase=phase,
                message="simulated transport failure",
            )
        ],
    )
    machine = PickAndPlaceStateMachine(
        injecting_sink,
        plan=accepted_plan(),
    )
    return machine, injecting_sink, recorder


@pytest.mark.parametrize("failed_command", SEQUENCE)
def test_each_state_machine_output_fault_transitions_to_failed(
    failed_command: str,
) -> None:
    machine, injecting_sink, recorder = make_machine(failed_command)

    machine.start_pick_and_place()
    success = machine.run_until_finished()

    failed_index = SEQUENCE.index(failed_command)

    assert not success
    assert machine.failed.is_active
    assert not machine.done.is_active
    assert machine.last_failed_command == failed_command
    assert machine.last_error is not None
    assert failed_command in machine.last_error
    assert "InjectedMotionFault" in machine.last_error
    assert "simulated transport failure" in machine.last_error

    assert [item.name for item in injecting_sink.attempted_commands] == (
        SEQUENCE[: failed_index + 1]
    )
    assert [item.name for item in recorder.commands] == SEQUENCE[:failed_index]


def test_after_send_fault_is_reported_as_failed_and_stops_sequence() -> None:
    machine, _, recorder = make_machine(
        "close_gripper",
        phase=FaultPhase.AFTER_SEND,
    )

    machine.start_pick_and_place()
    success = machine.run_until_finished()

    assert not success
    assert machine.failed.is_active
    assert machine.last_failed_command == "close_gripper"
    assert [item.name for item in recorder.commands] == SEQUENCE[:4]
    assert "after_send" in machine.last_error
    assert "lift_object" not in [item.name for item in recorder.commands]
