from dataclasses import dataclass
from typing import Protocol

from statemachine import StateMachine, State

from config.config_loader import load_config
from kinematics.angle_to_pwm import angle_to_pwm
from kinematics.inverse_kinematics import calculate_angles
from kinematics.workspace_checker import are_joint_angles_inside_limits


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

class DryRunMotionSink:
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

    def target_is_valid(self) -> bool:
        return self.target_validation_ok

    def on_enter_validating_target(self) -> None:
        target = self._require_target()

        result = calculate_angles(
            target.x_mm,
            target.y_mm,
            target.z_mm,
        )

        self.target_validation_ok = bool(result["reachable"])

        if not self.target_validation_ok:
            self.last_error = "; ".join(result["reasons"])

    def on_enter_moving_ready(self) -> None:
        self._send_named_pose("ready")

    def on_enter_moving_in_front_of_object(self) -> None:
        self._send_target_pose(
            "move_in_front_of_object",
            self._target_in_front_of_object()
        )

    def on_enter_advancing_towards_object(self) -> None:
        self._send_target_pose(
            "advance_towards_object",
            self._require_target()
        )

    def on_enter_closing_gripper(self) -> None:
        self._send_gripper_command("close_gripper", "closed_pulse_us")

    def on_enter_lifting_object(self) -> None:
        self._send_target_pose(
            "lift_object",
            self._target_in_front_of_object()
        )

    def on_enter_moving_to_deposit(self) -> None:
        self._send_named_pose("deposit")

    def on_enter_opening_gripper(self) -> None:
        self._send_gripper_command("open_gripper", "open_pulse_us")

    def on_enter_returning_home(self) -> None:
        self._send_named_pose("home")

    def on_enter_failed(self) -> None:
        if self.last_error is None:
            self.last_error = "State machine failed without a specific error message"

    def _send_target_pose(self, command_name: str, target: TargetPosition) -> None:
        result = calculate_angles(
            target.x_mm,
            target.y_mm,
            target.z_mm
        )

        if not result["reachable"]:
            self.last_error = "; ".join(result["reasons"])
            self.send("fail")
            return

        angles = result["angles_deg"]

        joint_angles = {
            "J1_base": angles["base"],
            "J2_shoulder": angles["base"],
            "J3_elbow": angles["base"],
        }

        pulses = self._joint_angles_to_pwm(joint_angles)

        self.sink.send(
            MotionCommand(
                name=command_name,
                pulses_us=pulses,
            )
        )

    def _send_named_pose(self, name: str) -> None:
        joint_angles = self._get_named_pose_angles(name)

        if not are_joint_angles_inside_limits(joint_angles):
            self.last_error = f"Named pose {name} not inside configured limits!"
            self.send("fail")
            return

        pulses = self._joint_angles_to_pwm(joint_angles)

        self.sink.send(
            MotionCommand(
                name=f"move_{name}",
                pulses_us=pulses,
            )
        )

    def _send_gripper_command(self, command_name: str, pulse_key: str) -> None:
        gripper_commands = self.poses_config["gripper_commands"]

        if pulse_key not in gripper_commands:
            self.last_error = f"Unknown gripper pulse key: {pulse_key}"
            self.send("fail")
            return

        pulse_us = int(gripper_commands[pulse_key])

        self.sink.send(
            MotionCommand(
                name=command_name,
                pulses_us={
                    "J5_gripper": pulse_us
                },
            )
        )

    def _joint_angles_to_pwm(self, joint_angles: dict[str, float]) -> dict[str, int]:
        joints = self.servo_calibration["joints"]
        pulses: dict[str, int] = {}

        for name, angle in joint_angles.items():
            if name not in joints:
                raise KeyError(f"Unknown joint name: {name}")

            pulses[name] = angle_to_pwm(angle, joints[name])

        return pulses

    def _get_named_pose_angles(self, name: str) -> dict[str, float]:
        poses = self.poses_config["poses"]

        if name not in poses:
            raise KeyError(f"Unknown pose: {name}")

        pose = poses[name]

        return {
            key: float(value) for key, value in pose.items() if key.startswith("J")
        }

    def _target_in_front_of_object(self) -> TargetPosition:
        ...

    def _target_lifted_from_object(self) -> TargetPosition:
        ...

    def _require_target(self) -> TargetPosition:
        if self.target is None:
            raise RuntimeError("No target position has been set")

        return self.target


if __name__ == "__main__":
    machine = PickAndPlaceStateMachine()

    machine.start_pick_and_place(
        TargetPosition(
            x_mm=200.0,
            y_mm=180.0,
            z_mm=60.0
        )
    )

    success = machine.run_until_finished()

    print(f"Final state: {machine.configuration}")
    print(f"Success: {success}")

    if machine.last_error is not None:
        print(f"Error: {machine.last_error}")