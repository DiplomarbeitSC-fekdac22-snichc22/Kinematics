import json
from pathlib import Path
from time import perf_counter
from typing import Protocol

from statemachine import State, StateMachine

from planning.models import (
    MotionCommand,
    MotionPlan,
    PlanningFailure,
    TargetPose,
    ValidationStatus,
)

# Backwards-compatible name used by existing API consumers.
TargetPosition = TargetPose


class MotionCommandSink(Protocol):
    def send(self, command: MotionCommand) -> None:
        ...


class DryRunMotionSink:
    """Intentional console-only sink that never accesses real hardware."""
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
    """Record commands and optionally forward them to another sink."""
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
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(
            json.dumps(self.commands, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return self.output_path


class PickAndPlaceStateMachine(StateMachine):
    idle = State("idle", initial=True)
    validating_plan = State("validating_plan")
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

    begin = idle.to(validating_plan)

    advance = (
        validating_plan.to(moving_ready, cond="plan_is_valid")
        | validating_plan.to(failed)
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
        validating_plan.to(failed)
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
        *,
        plan: MotionPlan,
    ) -> None:
        self.sink = sink or DryRunMotionSink()
        self.plan = plan
        self.plan_validation_ok = False
        self.last_error: str | None = None
        self.last_failed_command: str | None = None
        self.current_gripper_center_mm: dict[str, float] | None = None
        self._active_state_name: str | None = None
        self._state_started_at: float | None = None
        super().__init__()

    def start_pick_and_place(self) -> None:
        if not self.idle.is_active:
            raise RuntimeError(
                "Cannot start pick and place before idle "
                f"(current: {self.configuration})"
            )

        target = self.plan.target
        self.plan_validation_ok = False
        self.last_error = None
        self.last_failed_command = None
        self.current_gripper_center_mm = None
        self._active_state_name = None
        self._state_started_at = None

        print(
            "[ROBOT] Accepted plan target: "
            f"x={target.x_mm:.1f} mm, "
            f"y={target.y_mm:.1f} mm, "
            f"z={target.z_mm:.1f} mm"
        )
        self.send("begin")

    def run_until_finished(self) -> bool:
        while not self.done.is_active and not self.failed.is_active:
            self.send("advance")
        return self.done.is_active

    def plan_is_valid(self) -> bool:
        return self.plan_validation_ok

    def on_enter_validating_plan(self) -> None:
        self._announce_state("validating_plan", "Checking accepted motion plan")
        expected = (
            "ready",
            "pre_grasp",
            "grasp",
            "close_gripper",
            "lift",
            "retract",
            "deposit",
            "open_gripper",
            "home",
        )
        actual = tuple(waypoint.name for waypoint in self.plan.waypoints)
        statuses_valid = all(
            waypoint.validation_status is ValidationStatus.VALID
            for waypoint in self.plan.waypoints
        )
        self.plan_validation_ok = actual == expected and statuses_valid

        if self.plan_validation_ok:
            print("[ROBOT] Complete motion plan accepted")
        else:
            self.last_error = (
                "State machine received an incomplete or unvalidated plan: "
                f"expected {expected}, got {actual}"
            )

    def on_enter_moving_ready(self) -> None:
        self._enter_and_execute("moving_ready", "Moving to ready position", "ready")

    def on_enter_moving_in_front_of_object(self) -> None:
        self._enter_and_execute(
            "moving_in_front_of_object",
            "Moving in front of object",
            "pre_grasp",
        )

    def on_enter_advancing_towards_object(self) -> None:
        self._enter_and_execute(
            "advancing_towards_object",
            "Advancing towards object",
            "grasp",
        )

    def on_enter_closing_gripper(self) -> None:
        self._enter_and_execute(
            "closing_gripper",
            "Closing gripper",
            "close_gripper",
        )

    def on_enter_lifting_object(self) -> None:
        self._enter_and_execute("lifting_object", "Lifting object", "lift")

    def on_enter_retracting_from_shelf(self) -> None:
        self._enter_and_execute(
            "retracting_from_shelf",
            "Retracting from shelf before changing height",
            "retract",
        )

    def on_enter_moving_to_deposit(self) -> None:
        self._enter_and_execute(
            "moving_to_deposit",
            "Moving to deposit position",
            "deposit",
        )

    def on_enter_opening_gripper(self) -> None:
        self._enter_and_execute(
            "opening_gripper",
            "Opening gripper",
            "open_gripper",
        )

    def on_enter_returning_home(self) -> None:
        self._enter_and_execute("returning_home", "Returning home", "home")

    def on_enter_done(self) -> None:
        self._finish_state_timer()
        print("[ROBOT] Pick-and-place sequence completed")

    def on_enter_failed(self) -> None:
        self._finish_state_timer()
        if self.last_error is None:
            self.last_error = "State machine failed without a specific error message"
        print(f"[ROBOT] Pick-and-place sequence failed: {self.last_error}")

    def _enter_and_execute(
        self,
        state_name: str,
        message: str,
        waypoint_name: str,
    ) -> None:
        self._announce_state(state_name, message)
        self._send_planned_motion(waypoint_name)

    def _send_planned_motion(self, waypoint_name: str) -> None:
        try:
            command = self.plan.motion_for(waypoint_name).command
        except KeyError as exc:
            self.last_error = str(exc)
            self.send("fail")
            return

        if self._send_motion_command(command):
            self.current_gripper_center_mm = command.gripper_center_mm

    def _send_motion_command(self, command: MotionCommand) -> bool:
        try:
            self.sink.send(command)
        except Exception as exc:
            self.last_failed_command = command.name
            self.last_error = (
                f"Motion sink failed while executing {command.name!r}: "
                f"{type(exc).__name__}: {exc}"
            )
            self.send("fail")
            return False
        return True

    def _announce_state(self, state_name: str, message: str) -> None:
        self._finish_state_timer()
        self._active_state_name = state_name
        self._state_started_at = perf_counter()
        print(f"[ROBOT] {message}")

    def _finish_state_timer(self) -> None:
        if self._active_state_name is None or self._state_started_at is None:
            return
        elapsed_s = perf_counter() - self._state_started_at
        print(
            f"[ROBOT] State {self._active_state_name} "
            f"finished in {elapsed_s:.3f} s"
        )
        self._active_state_name = None
        self._state_started_at = None


if __name__ == "__main__":
    from planning.pick_and_place_planner import PickAndPlacePlanner

    target = TargetPose(x_mm=230.0, y_mm=180.0, z_mm=60.0)
    planner = PickAndPlacePlanner(enforce_hardware_safe_limits=False)
    plan_result = planner.plan(target)

    if isinstance(plan_result, PlanningFailure):
        print(f"[ROBOT] Planning failed: {plan_result.message}")
        raise SystemExit(1)

    for warning in plan_result.warnings:
        print(f"[ROBOT] {warning}")

    recorder = JsonRecordingMotionSink(
        output_path="pick_and_place_commands.json",
        wrapped_sink=DryRunMotionSink(),
    )
    machine = PickAndPlaceStateMachine(sink=recorder, plan=plan_result)
    machine.start_pick_and_place()
    success = machine.run_until_finished()
    output_path = recorder.export()

    print(f"[ROBOT] Final state: {machine.current_state.id}")
    print(f"[ROBOT] Success: {success}")
    print(f"[ROBOT] JSON written to {output_path.resolve()}")
    if machine.last_error is not None:
        print(f"[ROBOT] Error: {machine.last_error}")
