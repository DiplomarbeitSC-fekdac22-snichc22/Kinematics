from dataclasses import dataclass
from typing import Protocol

from statemachine import StateMachine, State


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

class PickAndPlaceStateMachine(StateMachine):
    idle = State("idle", initial=True)
    validating_target = State("validating_target")
    moving_ready = State("moving_ready")
    moving_in_front_of_object = State("moving_in_front_of_object")
    advancing_towards_object = State("advancing_towards_object")
    closing_gripper = State("closing_gripper")
    lift_gripper = State("lift_gripper")
    moving_to_deposit = State("moving_to_deposit")
    opening_gripper = State("opening_gripper")
    returning_home = State("returning_home")
    done = State("done", final=True)
    failed = State("failed", final=True)


