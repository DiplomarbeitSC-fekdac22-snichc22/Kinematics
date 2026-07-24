import argparse
import shlex
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from config.config_loader import DEFAULT_CONFIG_DIR, load_config
from hardware import Pca9685MotionSink
from motion import MotionExecutor
from planning.models import (
    MotionCommand,
    MotionPlan,
    PlannedMotion,
    PlanningFailure,
    TargetPose,
)
from planning.pick_and_place_planner import PickAndPlacePlanner
from planning.validators import validate_hardware_safe_pulses
from state_machine.pick_and_place import (
    DryRunMotionSink,
    PickAndPlaceStateMachine,
)


_CARTESIAN_ROLES = frozenset(
    {"theta1", "theta2", "theta3", "theta4"}
)


class ManualControlError(RuntimeError):
    """A user-correctable manual-control preflight failure"""


def _add_runtime_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=DEFAULT_CONFIG_DIR,
        help="Configuration directory (default: repository configs/)",
    )


def _add_motion_options(parser: argparse.ArgumentParser) -> None:
    _add_runtime_options(parser)
    parser.add_argument(
        "--hardware",
        action="store_true",
        help="Send the accepted command to the real PCA9685",
    )
    parser.add_argument(
        "--from-pose",
        help=(
            "Recorded pose that exactly matches the arm's physical starting "
            "position; required with --hardware"
        ),
    )


def _add_optional_coordinates(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("x_mm", type=float, nargs="?")
    parser.add_argument("y_mm", type=float, nargs="?")
    parser.add_argument("z_mm", type=float, nargs="?")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="robot-arm",
        description=(
            "Plan manual robot-arm commands and, after explicit preflight, "
            "send them to the physical PCA9685."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser(
        "check",
        help="Show physical-motion readiness without accessing hardware",
    )
    _add_runtime_options(check)

    pose = subparsers.add_parser(
        "pose",
        help="Plan or play back a physically recorded named pose",
    )
    pose.add_argument("name")
    _add_motion_options(pose)

    move = subparsers.add_parser(
        "move",
        help="Plan or execute one Cartesian gripper-centre move",
    )
    _add_optional_coordinates(move)
    _add_motion_options(move)

    pick = subparsers.add_parser(
        "pick",
        help="Plan or execute the existing pick-and-place sequence",
    )
    _add_optional_coordinates(pick)
    _add_motion_options(pick)

    control = subparsers.add_parser(
        "control",
        help=(
            "Keep the PCA9685 active and accept pose, XYZ, or exact PWM "
            "commands"
        ),
    )
    control.add_argument(
        "--hardware",
        action="store_true",
        help="Open the real PCA9685; otherwise print a dry-run session",
    )
    control.add_argument(
        "--home-pose",
        default="home",
        help="Recorded pose commanded when the session starts (default: home)",
    )
    _add_runtime_options(control)

    return parser


def _joint_names(servo_config: Mapping[str, Any]) -> tuple[str, ...]:
    joints = servo_config.get("joints")
    if not isinstance(joints, Mapping):
        raise ManualControlError(
            "servo_calibration.toml is missing the joints table"
        )
    return tuple(str(name) for name in joints)


def _load_recorded_pose(
    pose_name: str,
    *,
    config_dir: Path,
) -> tuple[dict[str, int], dict[str, float]]:
    poses_config = load_config("poses.toml", config_dir)
    servo_config = load_config("servo_calibration.toml", config_dir)
    pose = poses_config.get("poses", {}).get(pose_name)
    if not isinstance(pose, Mapping):
        raise ManualControlError(f"Unknown named pose {pose_name!r}")

    recorded = pose.get("recorded_pulses_us")
    if not isinstance(recorded, Mapping):
        raise ManualControlError(
            f"Pose {pose_name!r} has no physically recorded pulses"
        )

    expected_joints = _joint_names(servo_config)
    expected_joint_set = set(expected_joints)
    missing = tuple(sorted(expected_joint_set - set(recorded)))
    extra = tuple(sorted(set(recorded) - expected_joint_set))
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(f"missing {missing}")
        if extra:
            details.append(f"unknown {extra}")
        raise ManualControlError(
            f"Pose {pose_name!r} recorded pulses do not match configured "
            f"joints: {', '.join(details)}"
        )

    try:
        pulses = {
            joint_name: round(float(recorded[joint_name]))
            for joint_name in expected_joints
        }
        angles = {
            key: float(value)
            for key, value in pose.items()
            if str(key).startswith("J")
        }
    except (TypeError, ValueError) as exc:
        raise ManualControlError(
            f"Pose {pose_name!r} contains a non-numeric value"
        ) from exc

    pulse_reasons = validate_hardware_safe_pulses(
        pulses,
        servo_config,
    )
    if pulse_reasons:
        raise ManualControlError(
            f"Pose {pose_name!r} is outside the configured hardware-safe "
            f"range: {'; '.join(pulse_reasons)}"
        )

    return pulses, angles


def _recorded_pose_names(config_dir: Path) -> tuple[str, ...]:
    poses = load_config("poses.toml", config_dir).get("poses", {})
    if not isinstance(poses, Mapping):
        return ()
    return tuple(
        sorted(
            str(name)
            for name, pose in poses.items()
            if isinstance(pose, Mapping)
            and isinstance(pose.get("recorded_pulses_us"), Mapping)
        )
    )


def _cartesian_motion_blockers(
    servo_config: Mapping[str, Any],
) -> tuple[str, ...]:
    blockers: list[str] = []
    if not bool(
        servo_config.get("hardware_cartesian_motion_enabled", False)
    ):
        blockers.append(
            "servo_calibration.toml has "
            "hardware_cartesian_motion_enabled = false"
        )

    joints = servo_config.get("joints")
    if not isinstance(joints, Mapping):
        return tuple(
            blockers
            + ["servo_calibration.toml is missing the joints table"]
        )

    for joint_name, joint in joints.items():
        if not isinstance(joint, Mapping):
            blockers.append(f"{joint_name} has invalid calibration data")
            continue
        if (
            joint.get("kinematic_role") in _CARTESIAN_ROLES
            and bool(joint.get("requires_physical_calibration", True))
        ):
            blockers.append(
                f"{joint_name} still requires physical calibration"
            )
    return tuple(blockers)


def _resolve_target(
    args: argparse.Namespace,
    *,
    input_fn: Callable[[str], str],
) -> TargetPose:
    supplied = (args.x_mm, args.y_mm, args.z_mm)
    if all(value is None for value in supplied):
        try:
            return TargetPose(
                x_mm=float(input_fn("X forward/depth [mm]: ")),
                y_mm=float(input_fn("Y downward from top [mm]: ")),
                z_mm=float(input_fn("Z right/lateral [mm]: ")),
            )
        except ValueError as exc:
            raise ManualControlError(
                "Coordinates must be numeric"
            ) from exc
    if any(value is None for value in supplied):
        raise ManualControlError("Provide all three coordinates X Y Z, or omit all three to be prompted")
    return TargetPose(
        x_mm=float(args.x_mm),
        y_mm=float(args.y_mm),
        z_mm=float(args.z_mm),
    )


def _print_motion(command: MotionCommand) -> None:
    print(f"Command: {command.name}")
    if command.gripper_center_mm is not None:
        center = command.gripper_center_mm
        print(
            "Target:  "
            f"x={center['x_mm']:.1f} mm, "
            f"y={center['y_mm']:.1f} mm, "
            f"z={center['z_mm']:.1f} mm"
        )
    if command.joint_angles_deg is not None:
        angles = ", ".join(
            f"{name}={value:.2f} deg"
            for name, value in command.joint_angles_deg.items()
        )
        print(f"Angles:  {angles}")
    pulses = ", ".join(
        f"{name}={value} us"
        for name, value in command.pulses_us.items()
    )
    print(f"Pulses: {pulses}")


def _print_failure(failure: PlanningFailure) -> None:
    print(
        f"REJECTED [{failure.code}] at {failure.waypoint}: "
        f"{failure.message}",
        file=sys.stderr,
    )


def _confirm_hardware(
    *,
    action: str,
    start_pose: str,
    input_fn: Callable[[str], str],
) -> bool:
    print()
    print("REAL HARDWARE PREFLIGHT")
    print(f"Action: {action}")
    print(f"Declared physical start pose: {start_pose}")
    print("- Servo power is separate from Raspberry Pi power.")
    print("- The hardware emergency stop is reachable and cuts servo power.")
    print("- No person or loose object is inside the arm's motion envelope.")
    print("- The physical arm exactly matches the declared start pose.")
    return input_fn("Type MOVE to energize the servos: ").strip() == "MOVE"


def _confirm_control_session(
    *,
    home_pose: str,
    input_fn: Callable[[str], str],
) -> bool:
    print()
    print("REAL HARDWARE PREFLIGHT")
    print(f"Startup command: recorded pose {home_pose}")
    print("- The arm may move immediately when the startup PWM is applied.")
    print("- Servo power is separate from Raspberry Pi power.")
    print("- The hardware emergency stop is reachable and cuts servo power.")
    print("- No person or loose object is inside the arm's motion envelope.")
    return (
        input_fn("Type MOVE to command the startup pose: ").strip()
        == "MOVE"
    )


def _run_hardware_session(
    *,
    action_description: str,
    start_pose_name: str,
    start_pulses_us: dict[str, int],
    start_angles_deg: dict[str, float],
    config_dir: Path,
    action: Callable[[MotionExecutor], bool],
    return_to_start: bool,
    input_fn: Callable[[str], str],
) -> int:
    if not _confirm_hardware(
        action=action_description,
        start_pose=start_pose_name,
        input_fn=input_fn,
    ):
        print("Hardware command cancelled; no PCA9685 connection was opened.")
        return 2

    pca: Pca9685MotionSink | None = None
    try:
        pca = Pca9685MotionSink(config_dir=config_dir)
        executor = MotionExecutor(
            pca,
            initial_pulses_us=start_pulses_us,
            config_dir=config_dir,
        )
        start_command = MotionCommand(
            name=f"hold_{start_pose_name}",
            pulses_us=dict(start_pulses_us),
            joint_angles_deg=dict(start_angles_deg),
        )

        executor.send(start_command)
        success = action(executor)
        if not success:
            raise ManualControlError("Motion sequence reported failure")

        if return_to_start:
            input_fn(
                f"Target reached. Press Enter to return to "
                f"{start_pose_name}: "
            )
            executor.send(start_command)
            print(f"Returned to {start_pose_name}.")

        input_fn(
            "Switch OFF the separate servo power now, then press Enter to "
            "close the driver: "
        )
        return 0
    except KeyboardInterrupt:
        print(
            "\nInterrupted: disabling PCA9685 outputs. Use the hardware "
            "emergency stop if the arm is not safe.",
            file=sys.stderr,
        )
        return 130
    except Exception as exc:
        print(
            f"Hardware motion failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1
    finally:
        if pca is not None:
            try:
                pca.close()
            except Exception as exc:
                print(
                    f"Could not disable every PCA9685 output: {exc}",
                    file=sys.stderr,
                )


def _require_hardware_start_pose(
    args: argparse.Namespace,
) -> str:
    if not args.from_pose:
        raise ManualControlError(
            "--from-pose is required with --hardware because the arm has no "
            "joint feedback or homing switches"
        )
    return str(args.from_pose)


def _handle_check(args: argparse.Namespace) -> int:
    config_dir = Path(args.config_dir)
    servo_config = load_config("servo_calibration.toml", config_dir)
    pca_config = load_config("pca9685.toml", config_dir)
    blockers = _cartesian_motion_blockers(servo_config)

    print(
        "PCA9685: "
        f"address=0x{int(pca_config['i2c_address']):02x}, "
        f"frequency={float(pca_config['frequency_hz']):g} Hz"
    )
    print(f"Channels: {dict(pca_config['channel_map'])}")
    print(f"Recorded poses: {', '.join(_recorded_pose_names(config_dir))}")
    if blockers:
        print("Cartesian hardware motion: BLOCKED")
        for blocker in blockers:
            print(f"  - {blocker}")
        print("Recorded-pose playback is still available for commissioning.")
        return 2

    print("Cartesian hardware motion: configuration interlock is READY")
    print("This does not replace an I2C, power, E-stop, or clearance check.")
    return 0


def _handle_pose(
    args: argparse.Namespace,
    *,
    input_fn: Callable[[str], str],
) -> int:
    config_dir = Path(args.config_dir)
    pulses, angles = _load_recorded_pose(
        str(args.name),
        config_dir=config_dir,
    )
    target_command = MotionCommand(
        name=f"move_{args.name}",
        pulses_us=pulses,
        joint_angles_deg=angles,
    )
    _print_motion(target_command)

    if not args.hardware:
        print("DRY RUN: add --hardware and --from-pose to move the arm.")
        return 0

    start_pose = _require_hardware_start_pose(args)
    start_pulses, start_angles = _load_recorded_pose(
        start_pose,
        config_dir=config_dir,
    )
    return _run_hardware_session(
        action_description=f"move to recorded pose {args.name}",
        start_pose_name=start_pose,
        start_pulses_us=start_pulses,
        start_angles_deg=start_angles,
        config_dir=config_dir,
        action=lambda executor: (
            executor.send(target_command) is None
        ),
        return_to_start=True,
        input_fn=input_fn,
    )


def _plan_manual_move(
    *,
    target: TargetPose,
    start_pose: str,
    config_dir: Path,
    enforce_hardware_safe_limits: bool,
) -> PlannedMotion | PlanningFailure:
    _, start_angles = _load_recorded_pose(
        start_pose,
        config_dir=config_dir,
    )
    planner = PickAndPlacePlanner(
        config_dir=config_dir,
        enforce_hardware_safe_limits=enforce_hardware_safe_limits,
    )
    return planner.plan_cartesian_move(target, start_angles)


def _handle_move(
    args: argparse.Namespace,
    *,
    input_fn: Callable[[str], str],
) -> int:
    config_dir = Path(args.config_dir)
    target = _resolve_target(args, input_fn=input_fn)
    start_pose = (
        _require_hardware_start_pose(args)
        if args.hardware
        else str(args.from_pose or "home")
    )

    servo_config = load_config("servo_calibration.toml", config_dir)
    blockers = _cartesian_motion_blockers(servo_config)
    if args.hardware and blockers:
        raise ManualControlError(
            "Cartesian hardware motion is blocked: "
            + "; ".join(blockers)
        )

    result = _plan_manual_move(
        target=target,
        start_pose=start_pose,
        config_dir=config_dir,
        enforce_hardware_safe_limits=bool(args.hardware),
    )
    if isinstance(result, PlanningFailure):
        _print_failure(result)
        return 2

    for warning in result.waypoint.warnings:
        print(f"WARNING: {warning}")
    _print_motion(result.command)
    if not args.hardware:
        if blockers:
            print("DRY RUN ONLY: real XYZ motion is currently interlocked.")
            for blocker in blockers:
                print(f"  - {blocker}")
        return 0

    start_pulses, start_angles = _load_recorded_pose(
        start_pose,
        config_dir=config_dir,
    )
    return _run_hardware_session(
        action_description=(
            f"move to x={target.x_mm:.1f}, y={target.y_mm:.1f}, "
            f"z={target.z_mm:.1f} mm"
        ),
        start_pose_name=start_pose,
        start_pulses_us=start_pulses,
        start_angles_deg=start_angles,
        config_dir=config_dir,
        action=lambda executor: (
            executor.send(result.command) is None
        ),
        return_to_start=True,
        input_fn=input_fn,
    )


def _plan_pick(
    *,
    target: TargetPose,
    config_dir: Path,
    enforce_hardware_safe_limits: bool,
) -> MotionPlan | PlanningFailure:
    return PickAndPlacePlanner(
        config_dir=config_dir,
        enforce_hardware_safe_limits=enforce_hardware_safe_limits,
    ).plan(target)


def _run_plan(plan: MotionPlan, sink: Any) -> bool:
    machine = PickAndPlaceStateMachine(sink=sink, plan=plan)
    machine.start_pick_and_place()
    return machine.run_until_finished()


def _control_help(joint_names: tuple[str, ...]) -> None:
    print("Commands:")
    print("  home                    command the startup/home pose")
    print("  pose NAME               command a recorded pose exactly")
    print("  move X Y Z              command Cartesian coordinates in mm")
    print(
        "  pwm "
        + " ".join(joint_names)
        + "  command exact pulse widths in us"
    )
    print("  status                  show the last commanded PWM")
    print("  poses                   list recorded poses")
    print("  help                    show these commands")
    print("  release                 disable all outputs and exit")


def _parse_direct_pwm(
    values: Sequence[str],
    *,
    joint_names: tuple[str, ...],
    servo_config: Mapping[str, Any],
) -> dict[str, int]:
    if len(values) != len(joint_names):
        raise ManualControlError(
            f"pwm requires {len(joint_names)} integer pulse widths in this "
            f"order: {' '.join(joint_names)}"
        )
    try:
        pulses = {
            joint_name: int(value)
            for joint_name, value in zip(joint_names, values, strict=True)
        }
    except ValueError as exc:
        raise ManualControlError(
            "Every direct PWM value must be an integer number of microseconds"
        ) from exc

    reasons = validate_hardware_safe_pulses(pulses, servo_config)
    if reasons:
        raise ManualControlError(
            "Direct PWM is outside the configured hardware-safe range: "
            + "; ".join(reasons)
        )
    return pulses


def _print_control_status(
    *,
    command_name: str,
    pulses_us: Mapping[str, int],
) -> None:
    print(f"HOLDING: {command_name}")
    print(
        "Pulses: "
        + ", ".join(
            f"{joint_name}={pulse_us} us"
            for joint_name, pulse_us in pulses_us.items()
        )
    )


def _handle_control_line(
    line: str,
    *,
    config_dir: Path,
    home_pose_name: str,
    servo_config: Mapping[str, Any],
    joint_names: tuple[str, ...],
    send_command: Callable[[MotionCommand], None],
    current_command: MotionCommand,
    current_angles_deg: dict[str, float] | None,
) -> tuple[MotionCommand, dict[str, float] | None, bool]:
    try:
        tokens = shlex.split(line)
    except ValueError as exc:
        raise ManualControlError(f"Could not parse command: {exc}") from exc

    if not tokens:
        return current_command, current_angles_deg, False

    operation, *values = tokens
    operation = operation.lower()

    if operation in {"release", "quit", "exit"}:
        return current_command, current_angles_deg, True

    if operation == "help":
        _control_help(joint_names)
        return current_command, current_angles_deg, False

    if operation == "poses":
        print(
            "Recorded poses: "
            + ", ".join(_recorded_pose_names(config_dir))
        )
        return current_command, current_angles_deg, False

    if operation == "status":
        _print_control_status(
            command_name=current_command.name,
            pulses_us=current_command.pulses_us,
        )
        return current_command, current_angles_deg, False

    if operation == "home":
        if values:
            raise ManualControlError("home does not accept arguments")
        operation = "pose"
        values = [home_pose_name]

    if operation == "pose":
        if len(values) != 1:
            raise ManualControlError("pose requires exactly one pose name")
        pose_name = values[0]
        pulses, angles = _load_recorded_pose(
            pose_name,
            config_dir=config_dir,
        )
        command = MotionCommand(
            name=f"hold_pose_{pose_name}",
            pulses_us=pulses,
            joint_angles_deg=angles,
        )
        send_command(command)
        _print_control_status(
            command_name=command.name,
            pulses_us=command.pulses_us,
        )
        return command, angles, False

    if operation == "pwm":
        pulses = _parse_direct_pwm(
            values,
            joint_names=joint_names,
            servo_config=servo_config,
        )
        command = MotionCommand(
            name="hold_direct_pwm",
            pulses_us=pulses,
        )
        send_command(command)
        _print_control_status(
            command_name=command.name,
            pulses_us=command.pulses_us,
        )
        return command, None, False

    if operation == "move":
        if len(values) != 3:
            raise ManualControlError(
                "move requires X Y Z coordinates in millimetres"
            )
        if current_angles_deg is None:
            raise ManualControlError(
                "Cartesian movement is unavailable after a raw PWM command; "
                "command home or another recorded pose first"
            )
        blockers = _cartesian_motion_blockers(servo_config)
        if blockers:
            raise ManualControlError(
                "Cartesian hardware motion is blocked: "
                + "; ".join(blockers)
            )
        try:
            target = TargetPose(*(float(value) for value in values))
        except ValueError as exc:
            raise ManualControlError(
                "move coordinates must be numeric"
            ) from exc

        planner = PickAndPlacePlanner(
            config_dir=config_dir,
            enforce_hardware_safe_limits=True,
        )
        result = planner.plan_cartesian_move(target, current_angles_deg)
        if isinstance(result, PlanningFailure):
            raise ManualControlError(
                f"Cartesian target rejected [{result.code}] at "
                f"{result.waypoint}: {result.message}"
            )
        command = result.command
        send_command(command)
        _print_control_status(
            command_name=command.name,
            pulses_us=command.pulses_us,
        )
        return command, dict(command.joint_angles_deg or {}), False

    raise ManualControlError(
        f"Unknown control command {operation!r}; enter help"
    )


def _run_control_loop(
    *,
    config_dir: Path,
    home_pose_name: str,
    servo_config: Mapping[str, Any],
    joint_names: tuple[str, ...],
    send_command: Callable[[MotionCommand], None],
    home_command: MotionCommand,
    home_angles_deg: dict[str, float],
    input_fn: Callable[[str], str],
) -> None:
    current_command = home_command
    current_angles_deg: dict[str, float] | None = home_angles_deg

    send_command(home_command)
    _print_control_status(
        command_name=home_command.name,
        pulses_us=home_command.pulses_us,
    )
    print(
        "The PCA9685 will repeat this PWM until another command or release."
    )
    _control_help(joint_names)

    while True:
        try:
            line = input_fn("robot-arm> ")
        except EOFError:
            print()
            return

        try:
            current_command, current_angles_deg, should_release = (
                _handle_control_line(
                    line,
                    config_dir=config_dir,
                    home_pose_name=home_pose_name,
                    servo_config=servo_config,
                    joint_names=joint_names,
                    send_command=send_command,
                    current_command=current_command,
                    current_angles_deg=current_angles_deg,
                )
            )
        except (
            ManualControlError,
            KeyError,
            OSError,
            TypeError,
            ValueError,
        ) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            print("The previous PWM remains active.", file=sys.stderr)
            continue

        if should_release:
            return


def _handle_control(
    args: argparse.Namespace,
    *,
    input_fn: Callable[[str], str],
) -> int:
    config_dir = Path(args.config_dir)
    home_pose_name = str(args.home_pose)
    home_pulses, home_angles = _load_recorded_pose(
        home_pose_name,
        config_dir=config_dir,
    )
    servo_config = load_config("servo_calibration.toml", config_dir)
    joint_names = _joint_names(servo_config)
    home_command = MotionCommand(
        name=f"hold_pose_{home_pose_name}",
        pulses_us=home_pulses,
        joint_angles_deg=home_angles,
    )

    if not args.hardware:
        _run_control_loop(
            config_dir=config_dir,
            home_pose_name=home_pose_name,
            servo_config=servo_config,
            joint_names=joint_names,
            send_command=_print_motion,
            home_command=home_command,
            home_angles_deg=home_angles,
            input_fn=input_fn,
        )
        print("DRY RUN: no hardware connection was opened.")
        return 0

    if not _confirm_control_session(
        home_pose=home_pose_name,
        input_fn=input_fn,
    ):
        print("Hardware command cancelled; no PCA9685 connection was opened.")
        return 2

    pca: Pca9685MotionSink | None = None
    try:
        pca = Pca9685MotionSink(config_dir=config_dir)
        _run_control_loop(
            config_dir=config_dir,
            home_pose_name=home_pose_name,
            servo_config=servo_config,
            joint_names=joint_names,
            send_command=pca.send,
            home_command=home_command,
            home_angles_deg=home_angles,
            input_fn=input_fn,
        )
        print("Releasing all servo outputs.")
        return 0
    except KeyboardInterrupt:
        print(
            "\nInterrupted: releasing all servo outputs.",
            file=sys.stderr,
        )
        return 130
    except Exception as exc:
        print(
            f"Hardware control failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1
    finally:
        if pca is not None:
            try:
                pca.close()
            except Exception as exc:
                print(
                    f"Could not disable every PCA9685 output: {exc}",
                    file=sys.stderr,
                )


def _handle_pick(
    args: argparse.Namespace,
    *,
    input_fn: Callable[[str], str],
) -> int:
    config_dir = Path(args.config_dir)
    target = _resolve_target(args, input_fn=input_fn)
    servo_config = load_config("servo_calibration.toml", config_dir)
    blockers = _cartesian_motion_blockers(servo_config)
    if args.hardware and blockers:
        raise ManualControlError(
            "Pick-and-place hardware motion is blocked: "
            + "; ".join(blockers)
        )

    plan = _plan_pick(
        target=target,
        config_dir=config_dir,
        enforce_hardware_safe_limits=bool(args.hardware),
    )
    if isinstance(plan, PlanningFailure):
        _print_failure(plan)
        return 2
    for warning in plan.warnings:
        print(f"WARNING: {warning}")

    if not args.hardware:
        success = _run_plan(plan, DryRunMotionSink())
        if blockers:
            print("DRY RUN ONLY: real pick-and-place is interlocked.")
            for blocker in blockers:
                print(f"  - {blocker}")
        return 0 if success else 1

    start_pose = _require_hardware_start_pose(args)
    start_pulses, start_angles = _load_recorded_pose(
        start_pose,
        config_dir=config_dir,
    )
    return _run_hardware_session(
        action_description=(
            f"pick and place x={target.x_mm:.1f}, "
            f"y={target.y_mm:.1f}, z={target.z_mm:.1f} mm"
        ),
        start_pose_name=start_pose,
        start_pulses_us=start_pulses,
        start_angles_deg=start_angles,
        config_dir=config_dir,
        action=lambda executor: _run_plan(plan, executor),
        return_to_start=False,
        input_fn=input_fn,
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    input_fn: Callable[[str], str] = input,
) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "check":
            return _handle_check(args)
        if args.command == "pose":
            return _handle_pose(args, input_fn=input_fn)
        if args.command == "move":
            return _handle_move(args, input_fn=input_fn)
        if args.command == "pick":
            return _handle_pick(args, input_fn=input_fn)
        if args.command == "control":
            return _handle_control(args, input_fn=input_fn)
        raise ManualControlError(f"Unsupported command {args.command!r}")
    except (ManualControlError, KeyError, OSError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
