import pytest

from robot_testing.fault_injection import (
    FaultInjectingMotionSink,
    FaultPhase,
    FaultRule,
    InjectedMotionFault,
    RecordingMotionSink,
)
from state_machine.pick_and_place import MotionCommand


def command(name: str) -> MotionCommand:
    return MotionCommand(name=name, pulses_us={"J1_base": 1500})


def test_before_send_fault_does_not_deliver_command() -> None:
    recorder = RecordingMotionSink()
    sink = FaultInjectingMotionSink(
        recorder,
        [FaultRule(command_name="move_ready", message="I2C timeout")],
    )

    with pytest.raises(InjectedMotionFault, match="I2C timeout") as error:
        sink.send(command("move_ready"))

    assert error.value.occurrence.phase is FaultPhase.BEFORE_SEND
    assert [item.name for item in sink.attempted_commands] == ["move_ready"]
    assert sink.delivered_commands == []
    assert recorder.commands == []


def test_after_send_fault_models_ambiguous_delivery() -> None:
    recorder = RecordingMotionSink()
    sink = FaultInjectingMotionSink(
        recorder,
        [
            FaultRule(
                command_name="close_gripper",
                phase=FaultPhase.AFTER_SEND,
                message="acknowledgement lost",
            )
        ],
    )

    with pytest.raises(InjectedMotionFault, match="acknowledgement lost"):
        sink.send(command("close_gripper"))

    assert [item.name for item in sink.delivered_commands] == [
        "close_gripper"
    ]
    assert [item.name for item in recorder.commands] == ["close_gripper"]


def test_non_repeating_rule_is_consumed_after_first_match() -> None:
    recorder = RecordingMotionSink()
    sink = FaultInjectingMotionSink(
        recorder,
        [FaultRule(call_number=1, command_name="move_ready")],
    )

    with pytest.raises(InjectedMotionFault):
        sink.send(command("move_ready"))

    sink.send(command("move_ready"))

    assert sink.call_count == 2
    assert [item.name for item in recorder.commands] == ["move_ready"]


def test_rule_with_call_number_and_name_requires_both() -> None:
    recorder = RecordingMotionSink()
    sink = FaultInjectingMotionSink(
        recorder,
        [FaultRule(call_number=2, command_name="move_home")],
    )

    sink.send(command("move_home"))
    sink.send(command("move_ready"))
    sink.send(command("move_home"))

    assert sink.faults == []
    assert len(recorder.commands) == 3


def test_invalid_rule_is_rejected() -> None:
    with pytest.raises(ValueError, match="requires"):
        FaultRule()

    with pytest.raises(ValueError, match="at least 1"):
        FaultRule(call_number=0)
