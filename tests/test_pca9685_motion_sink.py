from dataclasses import dataclass
from pathlib import Path

import pytest

from hardware import Pca9685MotionSink

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


class FakeChannel:
    def __init__(self, duty_cycle: int = 0) -> None:
        self.duty_cycle = duty_cycle


class FakePca9685:
    def __init__(self) -> None:
        self.channels = [FakeChannel() for _ in range(16)]
        self.frequency: float | None = None


@dataclass(frozen=True)
class FakeMotionCommand:
    name: str
    pulses_us: dict[str, int]


def make_sink() -> tuple[Pca9685MotionSink, FakePca9685]:
    pca = FakePca9685()

    sink = Pca9685MotionSink(pca, config_dir=CONFIG_DIR)

    return sink, pca


def test_uses_existing_robot_controller_channel_mapping() -> None:
    sink, pca = make_sink()

    assert sink.channel_map == {
        "J1_base": 0,
        "J2_shoulder": 2,
        "J3_elbow": 4,
        "J4_wrist": 8,
        "J5_gripper": 6,
    }

    assert pca.frequency == 50


def test_send_converts_microseconds_and_writes_channels() -> None:
    sink, pca = make_sink()

    sink.send(FakeMotionCommand(
        name="move_and_grip",
        pulses_us={
            "J1_base": 1500,
            "J5_gripper": 1200,
        }
    ))

    # '<< 4' = 2⁴ = 16 -> e.g. 307 * 16 = 4912
    # We compute a 12-bit PCA9685 count (0..4095) and then
    # scale it to CircuitPython’s 16-bit duty-cycle space.
    assert pca.channels[0].duty_cycle == 307 << 4
    assert pca.channels[6].duty_cycle == 246 << 4

    # Unmentioned channels remain unchanged
    assert pca.channels[2].duty_cycle == 0


def test_invalid_pulse_does_not_particularly_update_outputs() -> None:
    sink, pca = make_sink()

    pca.channels[0].duty_cycle = 1234

    with pytest.raises(ValueError, match="configured safe range"):
        sink.send(FakeMotionCommand(
            name="unsafe",
            pulses_us={
                "J1_base": 1500,
                "J2_shoulder": 999,
            }
        ))

    # J1 was valid, but it must not have been written because J2 was invalid
    assert pca.channels[0].duty_cycle == 1234
    assert pca.channels[2].duty_cycle == 0


@pytest.mark.parametrize("pulse_us", [999, 2001])
def test_rejects_pulses_outside_safe_range(pulse_us: int) -> None:
    sink, _ = make_sink()

    with pytest.raises(ValueError, match="configured safe range"):
        sink.send(FakeMotionCommand(
            name="unsafe",
            pulses_us={
                "J3_elbow": pulse_us,
            }
        ))


def test_rejects_unknown_joint_names() -> None:
    sink, _ = make_sink()

    with pytest.raises(KeyError, match="Unknown servo output"):
        sink.send(FakeMotionCommand(
            name="unknown",
            pulses_us={
                "J6_unknown": 1500,
            }
        ))


def test_rejects_non_integer_pulses() -> None:
    sink, _ = make_sink()

    with pytest.raises(TypeError, match="integer of microseconds"):
        sink.send(FakeMotionCommand(
            name="wrong_type",
            pulses_us={
                "J1_base": 1500.0,
            }
        ))


def test_disable_all_clears_all_outputs() -> None:
    sink, pca = make_sink()

    for channel in pca.channels:
        channel.duty_cycle = 65535

    sink.disable_all()

    assert all(
        channel.duty_cycle == 0
        for channel in pca.channels
    )


def test_close_disables_outputs_and_prevents_commands() -> None:
    sink, pca = make_sink()

    pca.channels[0].duty_cycle = 4912

    sink.close()

    sink.close()

    assert all(
        channel.duty_cycle == 0
        for channel in pca.channels
    )

    with pytest.raises(RuntimeError, match="closed"):
        sink.send(FakeMotionCommand(
            name="after_close",
            pulses_us={
                "J1_base": 1500,
            }
        ))
