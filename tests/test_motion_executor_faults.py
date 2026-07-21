from pathlib import Path
from threading import Event

import pytest

from motion import MotionCancelled, MotionExecutionError, MotionExecutor
from robot_testing.fault_injection import (
    FaultInjectingMotionSink,
    FaultPhase,
    FaultRule,
    RecordingMotionSink,
)
from state_machine.pick_and_place import MotionCommand


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def motion_command(target_us: int = 1060) -> MotionCommand:
    return MotionCommand(
        name="move_test",
        pulses_us={"J1_base": target_us},
    )


def test_executor_preserves_last_acknowledged_frame_on_before_send_fault() -> None:
    recorder = RecordingMotionSink()
    injecting_sink = FaultInjectingMotionSink(
        recorder,
        [FaultRule(call_number=2, message="servo bus disconnected")],
    )
    executor = MotionExecutor(
        injecting_sink,
        initial_pulses_us={"J1_base": 1000},
        config_dir=CONFIG_DIR,
    )

    with pytest.raises(MotionExecutionError) as error:
        executor.send(motion_command())

    assert error.value.command_name == "move_test"
    assert error.value.frame_number == 2
    assert error.value.frame_count == 3
    assert error.value.pulses_us == {"J1_base": 1040}
    assert executor.last_pulses_us == {"J1_base": 1020}
    assert [item.pulses_us for item in recorder.commands] == [
        {"J1_base": 1020}
    ]


def test_executor_does_not_acknowledge_ambiguous_after_send_frame() -> None:
    recorder = RecordingMotionSink()
    injecting_sink = FaultInjectingMotionSink(
        recorder,
        [
            FaultRule(
                call_number=2,
                phase=FaultPhase.AFTER_SEND,
                message="write completed but acknowledgement was lost",
            )
        ],
    )
    executor = MotionExecutor(
        injecting_sink,
        initial_pulses_us={"J1_base": 1000},
        config_dir=CONFIG_DIR,
    )

    with pytest.raises(MotionExecutionError):
        executor.send(motion_command())

    assert executor.last_pulses_us == {"J1_base": 1020}
    assert [item.pulses_us for item in recorder.commands] == [
        {"J1_base": 1020},
        {"J1_base": 1040},
    ]


def test_pre_set_emergency_stop_sends_no_frames() -> None:
    recorder = RecordingMotionSink()
    emergency_stop = Event()
    emergency_stop.set()
    executor = MotionExecutor(
        recorder,
        emergency_stop=emergency_stop,
        initial_pulses_us={"J1_base": 1000},
        config_dir=CONFIG_DIR,
    )

    with pytest.raises(MotionCancelled, match="cancelled"):
        executor.send(motion_command())

    assert recorder.commands == []
    assert executor.last_pulses_us == {"J1_base": 1000}
