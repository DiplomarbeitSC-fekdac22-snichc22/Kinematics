from dataclasses import dataclass
from typing import Protocol

from statemachine import StateMachine, State

from config.config_loader import load_config


@dataclass(frozen=True)
class TargetPosition:
    x_mm: float
    y_mm: float
    z_mm: float

@dataclass(frozen=True)
class MotionCommand:
    name: str
    pulses_us: dict[str, int]

class MotionCommandSink(Protocol):
    def send(self, command: MotionCommand) -> None:
        ...

class DryRunMotionSink(Protocol):
    def send(self, command: MotionCommand) -> None:
        print(f"{command.name}: {command.pulses_us}")

class PickAndPlaceStateMachine(StateMachine):
    idle = State("idle", initial=True)
    validating_target = State("validating_target")
    moving_ready = State("moving_ready")
    moving_in_front_of_object = State("moving_in_front_of_object")
    advancing_towards_object = State("advancing_towards_object")
    closing_gripper = State("closing_gripper")
    lifting_object = State("lifting_object")
    moving_to_deposit = State("moving_to_deposit")
    opening_gripper = State("opening_gripper")
    returning_home = State("returning_home")
    done = State("done", final=True)
    failed = State("failed", final=True)

    begin = idle.to(validating_target)

    advance = (
            validating_target.to(moving_ready, cond="target_is_valid")
            | validating_target.to(failed)
            | moving_ready.to(moving_in_front_of_object)
            | moving_in_front_of_object.to(advancing_towards_object)
            | advancing_towards_object.to(closing_gripper)
            | closing_gripper.to(lifting_object)
            | lifting_object.to(moving_to_deposit)
            | moving_to_deposit.to(opening_gripper)
            | opening_gripper.to(returning_home)
            | returning_home.to(done)
    )

    fail = (
            validating_target.to(failed)
            | moving_ready.to(failed)
            | moving_in_front_of_object.to(failed)
            | advancing_towards_object.to(failed)
            | closing_gripper.to(failed)
            | lifting_object.to(failed)
            | moving_to_deposit.to(failed)
            | opening_gripper.to(failed)
            | returning_home.to(failed)
    )

    def __init__(self, sink: MotionCommandSink | None = None) -> None:
        self.sink = sink or DryRunMotionSink()

        self.target: TargetPosition | None = None
        self.target_validation_ok = False
        self.last_error: str | None = None

        self.kinematics_setting = load_config("kinematics_settings.toml")
        self.servo_calibration = load_config("servo_calibration.toml")
        self.poses_config = load_config("poses.toml")

        super().__init__()

    def start_pick_and_place(self, target: TargetPosition) -> None:
        if not self.idle.is_active:
            raise RuntimeError(f"Cannot start pick and place before idle (current: {self.configuration})")

        self.target = target
        self.target_validation_ok = False
        self.last_error = None

        self.send("begin")

    def run_until_finished(self) -> bool:
        while not self.done.is_active and not self.failed.is_active:
            self.send("advance")

        return self.done.is_active



