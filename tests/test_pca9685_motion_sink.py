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
        "J4_wrist": 6,
        "J5_gripper": 8,
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
    assert pca.channels[8].duty_cycle == 246 << 4

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