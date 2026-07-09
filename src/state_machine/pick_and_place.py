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
    lifting_object = State("lifting_object")
    moving_to_deposit = State("moving_to_deposit")
    opening_gripper = State("opening_gripper")
    returning_home = State("returning_home")
    done = State("done", final=True)
    failed = State("failed", final=True)

    begin = idle.to(validating_target)

    target_valid = validating_target.to(moving_ready)
    target_invalid = validating_target.to(failed)

    ready_reached = moving_ready.to(moving_in_front_of_object)
    above_object_reached = moving_in_front_of_object.to(advancing_towards_object)
    object_reached = advancing_towards_object.to(closing_gripper)
    gripper_closed = closing_gripper.to(lifting_object)
    object_lifted = lifting_object.to(moving_to_deposit)
    deposit_reached = moving_to_deposit.to(opening_gripper)
    gripper_opened = opening_gripper.to(returning_home)
    home_reached = returning_home.to(done)

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

