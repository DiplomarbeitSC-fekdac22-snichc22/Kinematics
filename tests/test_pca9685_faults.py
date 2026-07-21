from pathlib import Path

import pytest

from hardware import (
    Pca9685DisableError,
    Pca9685MotionSink,
    Pca9685WriteError,
)
from state_machine.pick_and_place import MotionCommand


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


class FaultingChannel:
    def __init__(self) -> None:
        self._duty_cycle = 0
        self.fail_nonzero_writes = False
        self.fail_zero_writes = False

    @property
    def duty_cycle(self) -> int:
        return self._duty_cycle

    @duty_cycle.setter
    def duty_cycle(self, value: int) -> None:
        if value == 0 and self.fail_zero_writes:
            raise OSError("zero write failed")
        if value != 0 and self.fail_nonzero_writes:
            raise OSError("PWM write failed")
        self._duty_cycle = value


class FaultingPca9685:
    def __init__(self) -> None:
        self.channels = [FaultingChannel() for _ in range(16)]
        self.frequency: float | None = None


def make_sink() -> tuple[Pca9685MotionSink, FaultingPca9685]:
    pca = FaultingPca9685()
    return Pca9685MotionSink(pca, config_dir=CONFIG_DIR), pca


def test_channel_write_failure_attempts_to_disable_every_output() -> None:
    sink, pca = make_sink()
    pca.channels[2].fail_nonzero_writes = True

    with pytest.raises(Pca9685WriteError) as error:
        sink.send(
            MotionCommand(
                name="move_two_joints",
                pulses_us={
                    "J1_base": 1500,
                    "J2_shoulder": 1500,
                },
            )
        )

    assert error.value.command_name == "move_two_joints"
    assert error.value.joint_name == "J2_shoulder"
    assert error.value.channel == 2
    assert error.value.disable_failures == ()
    assert all(channel.duty_cycle == 0 for channel in pca.channels)


def test_disable_all_continues_after_individual_channel_failure() -> None:
    sink, pca = make_sink()
    for channel in pca.channels:
        channel._duty_cycle = 65535
    pca.channels[4].fail_zero_writes = True

    with pytest.raises(Pca9685DisableError) as error:
        sink.disable_all()

    assert error.value.failed_channels == (4,)
    assert pca.channels[4].duty_cycle == 65535
    assert all(
        channel.duty_cycle == 0
        for index, channel in enumerate(pca.channels)
        if index != 4
    )
