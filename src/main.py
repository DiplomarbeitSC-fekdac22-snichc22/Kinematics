import argparse
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
        raise ManualControlError(f"Unsupported command {args.command!r}")
    except (ManualControlError, KeyError, OSError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
