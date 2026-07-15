import json
from dataclasses import dataclass
from math import hypot
from pathlib import Path
from time import perf_counter
from typing import Protocol

from statemachine import StateMachine, State

from config.config_loader import load_config
from kinematics.angle_to_pwm import angle_to_pwm
from kinematics.forward_kinematics import calculate_gripper_center
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
    gripper_center_mm: dict[str, float] | None = None
    joint_angles_deg: dict[str, float] | None = None


class MotionCommandSink(Protocol):
    def send(self, command: MotionCommand) -> None:
        ...


class DryRunMotionSink:
    """Intentional console-only sink.

    This sink prints generated commands but never communicates with
    physical robot hardware.
    """

    def send(self, command: MotionCommand) -> None:
        if command.gripper_center_mm is None:
            coordinates = "unknown"
        else:
            position = command.gripper_center_mm
            coordinates = (
                f"x={position['x_mm']:.1f} mm, "
                f"y={position['y_mm']:.1f} mm, "
                f"z={position['z_mm']:.1f} mm"
            )

        print(
            f"[ROBOT] Command generated: {command.name} | "
            f"Gripper center: {coordinates} | "
            f"PWM: {command.pulses_us}"
        )


class JsonRecordingMotionSink:
    """Records commands and optionally forwards them to another sink."""

    def __init__(
            self,
            output_path: str | Path,
            wrapped_sink: MotionCommandSink | None = None,
    ) -> None:
        self.output_path = Path(output_path)
        self.wrapped_sink = wrapped_sink
        self.commands: list[dict[str, object]] = []

    def send(self, command: MotionCommand) -> None:
        if self.wrapped_sink is not None:
            self.wrapped_sink.send(command)

        self.commands.append(
            {
                "command": command.name,
                "pulses_us": dict(command.pulses_us),
                "joint_angles_deg": (
                    dict(command.joint_angles_deg)
                    if command.joint_angles_deg is not None
                    else None
                ),
                "gripper_center_mm": (
                    dict(command.gripper_center_mm)
                    if command.gripper_center_mm is not None
                    else None
                ),
            }
        )

    def export(self) -> Path:
        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.output_path.write_text(
            json.dumps(
                self.commands,
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

        return self.output_path


class PickAndPlaceStateMachine(StateMachine):
    idle = State("idle", initial=True)
    validating_target = State("validating_target")
    moving_ready = State("moving_ready")
    moving_in_front_of_object = State("moving_in_front_of_object")
    advancing_towards_object = State("advancing_towards_object")
    closing_gripper = State("closing_gripper")
    lifting_object = State("lifting_object")
    retracting_from_shelf = State("retracting_from_shelf")
    moving_to_deposit = State("moving_to_deposit")
    opening_gripper = State("opening_gripper")
    returning_home = State("returning_home")
    done = State("done", final=True)
    failed = State("failed", final=True)

    begin = idle.to(validating_target)

    advance = (
            validating_target.to(
                moving_ready,
                cond="target_is_valid",
            )
            | validating_target.to(failed)
            | moving_ready.to(moving_in_front_of_object)
            | moving_in_front_of_object.to(advancing_towards_object)
            | advancing_towards_object.to(closing_gripper)
            | closing_gripper.to(lifting_object)
            | lifting_object.to(retracting_from_shelf)
            | retracting_from_shelf.to(moving_to_deposit)
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
            | retracting_from_shelf.to(failed)
            | moving_to_deposit.to(failed)
            | opening_gripper.to(failed)
            | returning_home.to(failed)
    )

    def __init__(
            self,
            sink: MotionCommandSink | None = None,
    ) -> None:
        self.sink = sink or DryRunMotionSink()

        self.target: TargetPosition | None = None
        self.target_validation_ok = False
        self.last_error: str | None = None

        self.current_gripper_center_mm: dict[str, float] | None = None

        self._active_state_name: str | None = None
        self._state_started_at: float | None = None

        self.kinematics_setting = load_config(
            "kinematics_settings.toml"
        )
        self.servo_calibration = load_config(
            "servo_calibration.toml"
        )
        self.poses_config = load_config(
            "poses.toml"
        )

        super().__init__()

    def start_pick_and_place(
            self,
            target: TargetPosition,
    ) -> None:
        if not self.idle.is_active:
            raise RuntimeError(
                "Cannot start pick and place before idle "
                f"(current: {self.configuration})"
            )

        self.target = target
        self.target_validation_ok = False
        self.last_error = None

        self.current_gripper_center_mm = None
        self._active_state_name = None
        self._state_started_at = None

        print(
            "[ROBOT] Target received: "
            f"x={target.x_mm:.1f} mm, "
            f"y={target.y_mm:.1f} mm, "
            f"z={target.z_mm:.1f} mm"
        )

        self.send("begin")

    def run_until_finished(self) -> bool:
        while (
                not self.done.is_active
                and not self.failed.is_active
        ):
            self.send("advance")

        return self.done.is_active

    def target_is_valid(self) -> bool:
        return self.target_validation_ok

    def on_enter_validating_target(self) -> None:
        self._announce_state(
            "validating_target",
            "Validating target",
        )

        target = self._require_target()

        result = calculate_angles(
            target.x_mm,
            target.y_mm,
            target.z_mm,
        )

        self.target_validation_ok = bool(
            result["reachable"]
        )

        if self.target_validation_ok:
            print("[ROBOT] Target accepted")
        else:
            self.last_error = "; ".join(
                result["reasons"]
            )

            print(
                f"[ROBOT] Target rejected: {self.last_error}"
            )

    def on_enter_moving_ready(self) -> None:
        self._announce_state(
            "moving_ready",
            "Moving to ready position",
        )

        self._send_named_pose("ready")

    def on_enter_moving_in_front_of_object(self) -> None:
        self._announce_state(
            "moving_in_front_of_object",
            "Moving in front of object",
        )

        self._send_target_pose(
            "move_in_front_of_object",
            self._target_in_front_of_object(),
        )

    def on_enter_advancing_towards_object(self) -> None:
        self._announce_state(
            "advancing_towards_object",
            "Advancing towards object",
        )

        self._send_target_pose(
            "advance_towards_object",
            self._target_at_grasp_depth(),
        )

    def on_enter_closing_gripper(self) -> None:
        self._announce_state(
            "closing_gripper",
            "Closing gripper",
        )

        self._send_gripper_command(
            "close_gripper",
            "closed_pulse_us",
        )

    def on_enter_lifting_object(self) -> None:
        self._announce_state(
            "lifting_object",
            "Lifting object",
        )

        self._send_target_pose(
            "lift_object",
            self._target_lifted_from_object(),
        )

    def on_enter_retracting_from_shelf(self) -> None:
        self._announce_state(
            "retracting_from_shelf",
            "Retracting from shelf before changing height",
        )

        self._send_target_pose(
            "retract_from_shelf",
            self._target_retracted_from_shelf(),
        )

    def on_enter_moving_to_deposit(self) -> None:
        self._announce_state(
            "moving_to_deposit",
            "Moving to deposit position",
        )

        drop_off = self.poses_config[
            "cartesian_targets"
        ]["drop_off"]

        self._send_target_pose(
            "move_deposit",
            TargetPosition(
                x_mm=float(drop_off["x_mm"]),
                y_mm=float(drop_off["y_mm"]),
                z_mm=float(drop_off["z_mm"]),
            ),
        )

    def on_enter_opening_gripper(self) -> None:
        self._announce_state(
            "opening_gripper",
            "Opening gripper",
        )

        self._send_gripper_command(
            "open_gripper",
            "open_pulse_us",
        )

    def on_enter_returning_home(self) -> None:
        self._announce_state(
            "returning_home",
            "Returning home",
        )

        self._send_named_pose("home")

    def on_enter_done(self) -> None:
        self._finish_state_timer()

        print(
            "[ROBOT] Pick-and-place sequence completed"
        )

    def on_enter_failed(self) -> None:
        self._finish_state_timer()

        if self.last_error is None:
            self.last_error = (
                "State machine failed without a "
                "specific error message"
            )

        print(
            "[ROBOT] Pick-and-place sequence failed: "
            f"{self.last_error}"
        )

    def _announce_state(
            self,
            state_name: str,
            message: str,
    ) -> None:
        self._finish_state_timer()

        self._active_state_name = state_name
        self._state_started_at = perf_counter()

        print(f"[ROBOT] {message}")

    def _finish_state_timer(self) -> None:
        if (
                self._active_state_name is None
                or self._state_started_at is None
        ):
            return

        elapsed_s = (
                perf_counter() - self._state_started_at
        )

        print(
            f"[ROBOT] State {self._active_state_name} "
            f"finished in {elapsed_s:.3f} s"
        )

        self._active_state_name = None
        self._state_started_at = None

    def _send_target_pose(
            self,
            command_name: str,
            target: TargetPosition,
    ) -> None:
        result = calculate_angles(
            target.x_mm,
            target.y_mm,
            target.z_mm,
        )

        if not result["reachable"]:
            self.last_error = "; ".join(
                result["reasons"]
            )

            self.send("fail")
            return

        angles = result["angles_deg"]

        joint_angles = {
            "J1_base": float(angles["base"]),
            "J2_shoulder": float(angles["shoulder"]),
            "J3_elbow": float(angles["elbow"]),
            "J4_wrist": float(angles["wrist"]),
        }

        pulses = self._joint_angles_to_pwm(
            joint_angles
        )

        gripper_center = calculate_gripper_center(
            joint_angles
        )

        self.current_gripper_center_mm = (
            gripper_center
        )

        self.sink.send(
            MotionCommand(
                name=command_name,
                pulses_us=pulses,
                joint_angles_deg=joint_angles,
                gripper_center_mm=gripper_center,
            )
        )

    def _send_named_pose(
            self,
            name: str,
    ) -> None:
        joint_angles = self._get_named_pose_angles(
            name
        )

        if not are_joint_angles_inside_limits(
                joint_angles
        ):
            self.last_error = (
                f"Named pose {name} not inside "
                "configured limits!"
            )

            self.send("fail")
            return

        pulses = self._joint_angles_to_pwm(
            joint_angles
        )

        # Named ready/home poses describe arm angles. J5_gripper=0 is only a
        # placeholder and previously converted to the servo-neutral 1500 us,
        # which is neither the configured open nor closed state. Keep the
        # gripper explicitly open while moving to ready or home.
        pulses["J5_gripper"] = int(
            self.poses_config["gripper_commands"]["open_pulse_us"]
        )

        gripper_center = calculate_gripper_center(
            joint_angles
        )

        self.current_gripper_center_mm = (
            gripper_center
        )

        self.sink.send(
            MotionCommand(
                name=f"move_{name}",
                pulses_us=pulses,
                joint_angles_deg=joint_angles,
                gripper_center_mm=gripper_center,
            )
        )

    def _send_gripper_command(
            self,
            command_name: str,
            pulse_key: str,
    ) -> None:
        gripper_commands = self.poses_config[
            "gripper_commands"
        ]

        if pulse_key not in gripper_commands:
            self.last_error = (
                f"Unknown gripper pulse key: "
                f"{pulse_key}"
            )

            self.send("fail")
            return

        pulse_us = int(
            gripper_commands[pulse_key]
        )

        self.sink.send(
            MotionCommand(
                name=command_name,
                pulses_us={
                    "J5_gripper": pulse_us
                },
                gripper_center_mm=(
                    self.current_gripper_center_mm
                ),
            )
        )

    def _joint_angles_to_pwm(
            self,
            joint_angles: dict[str, float],
    ) -> dict[str, int]:
        joints = self.servo_calibration["joints"]
        pulses: dict[str, int] = {}

        for name, angle in joint_angles.items():
            if name not in joints:
                raise KeyError(
                    f"Unknown joint name: {name}"
                )

            pulses[name] = angle_to_pwm(
                angle,
                joints[name],
            )

        return pulses

    def _get_named_pose_angles(
            self,
            name: str,
    ) -> dict[str, float]:
        poses = self.poses_config["poses"]

        if name not in poses:
            raise KeyError(
                f"Unknown pose: {name}"
            )

        pose = poses[name]

        return {
            key: float(value)
            for key, value in pose.items()
            if key.startswith("J")
        }

    def _target_in_front_of_object(
            self,
    ) -> TargetPosition:
        target = self._require_target()
        offsets = self.kinematics_setting["target_offsets"]
        return self._target_with_radial_retraction(
            target,
            y_mm=(
                target.y_mm
                - float(offsets["pre_grasp_y_offset_mm"])
            ),
            retraction_mm=(
                    float(offsets["grasp_depth_offset_mm"])
                    + float(offsets["approach_r_offset_mm"])
            ),
        )

    def _target_retracted_from_shelf(
            self,
    ) -> TargetPosition:
        target = self._require_target()
        offsets = self.kinematics_setting["target_offsets"]
        return self._target_with_radial_retraction(
            target,
            y_mm=(
                target.y_mm
                - float(offsets["lift_after_grip_y_offset_mm"])
            ),
            retraction_mm=(
                    float(offsets["grasp_depth_offset_mm"])
                    + float(offsets["approach_r_offset_mm"])
            ),
        )

    def _target_with_radial_retraction(
            self,
            target: TargetPosition,
            *,
            y_mm: float,
            retraction_mm: float,
    ) -> TargetPosition:
        base_x, _, base_z = (
            self.kinematics_setting[
                "input_coordinates"
            ][
                "base_rotation_axis_at_"
                "mounting_plate_mm"
            ]
        )

        delta_x = target.x_mm - base_x
        delta_z = target.z_mm - base_z

        radial_distance = hypot(
            delta_x,
            delta_z,
        )

        if retraction_mm < 0.0:
            raise ValueError("Radial retraction cannot be negative")

        if radial_distance <= retraction_mm:
            raise ValueError(
                "Target is too close to apply the "
                "configured radial retraction"
            )

        scale = (
                        radial_distance - retraction_mm
                ) / radial_distance

        return TargetPosition(
            x_mm=base_x + delta_x * scale,
            y_mm=y_mm,
            z_mm=base_z + delta_z * scale,
        )

    def _target_lifted_from_object(
            self,
    ) -> TargetPosition:
        target = self._require_target()
        offsets = self.kinematics_setting["target_offsets"]
        return self._target_with_radial_retraction(
            target,
            y_mm=(
                target.y_mm
                - float(offsets["lift_after_grip_y_offset_mm"])
            ),
            retraction_mm=float(offsets["grasp_depth_offset_mm"]),
        )

    def _target_at_grasp_depth(self) -> TargetPosition:
        """Align the jaw contact centre with the object centre."""
        target = self._require_target()
        return self._target_with_radial_retraction(
            target,
            y_mm=target.y_mm,
            retraction_mm=float(
                self.kinematics_setting[
                    "target_offsets"
                ]["grasp_depth_offset_mm"]
            ),
        )

    def _require_target(self) -> TargetPosition:
        if self.target is None:
            raise RuntimeError(
                "No target position has been set"
            )

        return self.target


if __name__ == "__main__":
    recorder = JsonRecordingMotionSink(
        output_path="pick_and_place_commands.json",
        wrapped_sink=DryRunMotionSink(),
    )

    machine = PickAndPlaceStateMachine(
        sink=recorder
    )

    machine.start_pick_and_place(
        TargetPosition(
            x_mm=230.0,
            y_mm=180.0,
            z_mm=60.0,
        )
    )

    success = machine.run_until_finished()
    output_path = recorder.export()

    print(
        f"[ROBOT] Final state: "
        f"{machine.configuration}"
    )
    print(f"[ROBOT] Success: {success}")
    print(
        f"[ROBOT] JSON written to "
        f"{output_path.resolve()}"
    )

    if machine.last_error is not None:
        print(
            f"[ROBOT] Error: "
            f"{machine.last_error}"
        )
