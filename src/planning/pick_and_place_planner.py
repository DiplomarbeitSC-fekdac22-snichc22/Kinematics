from dataclasses import replace
from pathlib import Path
from typing import Any

from config.config_loader import DEFAULT_CONFIG_DIR, load_config
from kinematics.forward_kinematics import calculate_gripper_center
from kinematics.inverse_kinematics import calculate_angles
from planning.models import (
    MotionCommand,
    MotionPlan,
    PlannedMotion,
    PlanningFailure,
    TargetPose,
    ValidationStatus,
    Waypoint,
)
from planning.validators import (
    convert_joint_angles_to_pwm,
    validate_gripper_pulse,
    validate_hardware_safe_pulses,
    validate_joint_angles,
    validate_workspace,
    validate_xyz_values,
)
from planning.waypoint_generator import (
    WaypointGenerationError,
    WaypointGenerator,
)


_IK_RESULT_KEYS_BY_ROLE = {
    "theta1": "base",
    "theta2": "shoulder",
    "theta3": "elbow",
    "theta4": "wrist",
}


def _failure(
        waypoint: Waypoint,
    code: str,
    reasons: tuple[str, ...],
) -> PlanningFailure:
    rejected = replace(
        waypoint,
        validation_status=ValidationStatus.INVALID,
        rejection_reasons=reasons,
    )
    message = f"Waypoint {waypoint.name!r} rejected: {'; '.join(reasons)}"
    return PlanningFailure(
        waypoint=waypoint.name,
        code=code,
        message=message,
        reasons=reasons,
        rejected_waypoint=rejected,
    )


class PickAndPlacePlanner:
    """Generate every command and reject the sequence as one transaction."""
    def __init__(
        self,
        *,
        config_dir: Path | str = DEFAULT_CONFIG_DIR,
        enforce_hardware_safe_limits: bool = True,
    ) -> None:
        self.config_dir = Path(config_dir)
        self.enforce_hardware_safe_limits = enforce_hardware_safe_limits
        self.kinematics_settings = load_config(
            "kinematics_settings.toml",
            self.config_dir,
        )
        self.servo_calibration = load_config(
            "servo_calibration.toml",
            self.config_dir,
        )
        self.poses = load_config(
            "poses.toml",
            self.config_dir,
        )
        self.generator = WaypointGenerator(
            config_dir=self.config_dir,
            kinematics_settings=self.kinematics_settings,
            poses_config=self.poses,
        )

    def plan(self, target: TargetPose) -> MotionPlan | PlanningFailure:
        """Return a complete plan or one structured rejection."""
        input_reasons = validate_xyz_values(target)
        if input_reasons:
            return _failure(
                Waypoint(
                    name="grasp",
                    command_name="advance_towards_object",
                    cartesian_target=target,
                ),
                "INVALID_CARTESIAN_TARGET",
                input_reasons,
            )

        try:
            generated = self.generator.generate(target)
        except WaypointGenerationError as exc:
            waypoint = Waypoint(
                name=exc.waypoint,
                command_name=exc.waypoint,
            )
            return _failure(
                waypoint,
                "WAYPOINT_GENERATION_ERROR",
                (str(exc),),
            )
        except (KeyError, TypeError, ValueError) as exc:
            waypoint = Waypoint(
                name="plan",
                command_name="plan",
            )
            return _failure(
                waypoint,
                "CONFIGURATION_ERROR",
                (f"{type(exc).__name__}: {exc}",),
            )

        planned: list[PlannedMotion] = []
        plan_warnings: list[str] = []
        for waypoint in generated:
            result = self._validate_waypoint(waypoint)
            if isinstance(result, PlanningFailure):
                return result
            planned.append(result)
            plan_warnings.extend(result.waypoint.warnings)

        return MotionPlan(
            target=target,
            motions=tuple(planned),
            warnings=tuple(plan_warnings),
        )

    def _validate_waypoint(
        self,
        waypoint: Waypoint,
    ) -> PlannedMotion | PlanningFailure:
        if waypoint.gripper_pulse_key is not None:
            return self._validate_gripper_waypoint(waypoint)
        if waypoint.named_pose is not None:
            return self._validate_named_pose_waypoint(waypoint)
        if waypoint.cartesian_target is not None:
            return self._validate_cartesian_waypoint(waypoint)
        return _failure(
            waypoint,
            "INVALID_WAYPOINT",
            ("Waypoint has no Cartesian target, named pose, or gripper action",),
        )

    def _validate_cartesian_waypoint(
        self,
        waypoint: Waypoint,
    ) -> PlannedMotion | PlanningFailure:
        target = waypoint.cartesian_target
        assert target is not None

        xyz_reasons = validate_xyz_values(target)
        if xyz_reasons:
            return _failure(
                waypoint,
                "INVALID_CARTESIAN_TARGET",
                xyz_reasons,
            )

        workspace_reasons = validate_workspace(
            target,
            self.kinematics_settings,
        )
        if workspace_reasons:
            return _failure(
                waypoint,
                "WORKSPACE_VIOLATION",
                workspace_reasons,
            )

        try:
            ik_result = calculate_angles(
                target.x_mm,
                target.y_mm,
                target.z_mm,
                self.config_dir,
            )
        except (ArithmeticError, KeyError, TypeError, ValueError) as exc:
            return _failure(
                waypoint,
                "IK_SOLVER_ERROR",
                (f"{type(exc).__name__}: {exc}",),
            )

        joint_angles = self._joint_angles_from_ik(ik_result)
        joint_reasons = validate_joint_angles(
            joint_angles,
            self.servo_calibration,
        )
        pulse_values, pulse_reasons = convert_joint_angles_to_pwm(
            joint_angles,
            self.servo_calibration,
        )

        ik_reasons = tuple(
            ik_result.get("reason_groups", {}).get("geometry", ())
        )
        if ik_reasons:
            return _failure(
                waypoint,
                "IK_UNREACHABLE",
                ik_reasons,
            )
        if joint_reasons:
            return _failure(
                waypoint,
                "JOINT_LIMIT_VIOLATION",
                joint_reasons,
            )
        if pulse_reasons:
            return _failure(
                waypoint,
                "SERVO_PULSE_LIMIT_VIOLATION",
                pulse_reasons,
            )

        return self._accepted_motion(
            waypoint,
            joint_angles=joint_angles,
            pulses_us=pulse_values,
            gripper_center=target.as_dict(),
            ik_branch=str(
                self.kinematics_settings["ik"]["solution_preference"]
            ),
        )

    def _validate_named_pose_waypoint(
        self,
        waypoint: Waypoint,
    ) -> PlannedMotion | PlanningFailure:
        pose_name = waypoint.named_pose
        assert pose_name is not None
        pose = self.poses.get("poses", {}).get(pose_name)
        if not isinstance(pose, dict):
            return _failure(
                waypoint,
                "UNKNOWN_NAMED_POSE",
                (f"Unknown named pose {pose_name!r}",),
            )

        try:
            joint_angles = {
                key: float(value)
                for key, value in pose.items()
                if key.startswith("J")
            }
        except (TypeError, ValueError) as exc:
            return _failure(
                waypoint,
                "INVALID_NAMED_POSE",
                (f"{type(exc).__name__}: {exc}",),
            )

        required_joints = {
            joint_name
            for joint_name, joint in self.servo_calibration["joints"].items()
            if joint.get("kinematic_role") in _IK_RESULT_KEYS_BY_ROLE
        }
        required_joints.add(
            str(self.kinematics_settings["model"]["gripper_joint"])
        )
        missing_joints = tuple(sorted(required_joints - joint_angles.keys()))
        if missing_joints:
            return _failure(
                waypoint,
                "INVALID_NAMED_POSE",
                (f"Named pose is missing joints: {missing_joints}",),
            )

        joint_reasons = validate_joint_angles(
            joint_angles,
            self.servo_calibration,
        )
        if joint_reasons:
            return _failure(
                waypoint,
                "JOINT_LIMIT_VIOLATION",
                joint_reasons,
            )

        pulses, pulse_reasons = convert_joint_angles_to_pwm(
            joint_angles,
            self.servo_calibration,
        )
        if pulse_reasons:
            return _failure(
                waypoint,
                "SERVO_PULSE_LIMIT_VIOLATION",
                pulse_reasons,
            )

        gripper_joint = str(
            self.kinematics_settings["model"]["gripper_joint"]
        )
        open_pulse = int(
            self.poses["gripper_commands"]["open_pulse_us"]
        )
        gripper_reasons = validate_gripper_pulse(
            gripper_joint,
            open_pulse,
            self.servo_calibration,
        )
        if gripper_reasons:
            return _failure(
                waypoint,
                "SERVO_PULSE_LIMIT_VIOLATION",
                gripper_reasons,
            )
        pulses[gripper_joint] = open_pulse

        try:
            gripper_center = calculate_gripper_center(
                joint_angles,
                self.config_dir,
            )
        except (ArithmeticError, KeyError, TypeError, ValueError) as exc:
            return _failure(
                waypoint,
                "FORWARD_KINEMATICS_ERROR",
                (f"{type(exc).__name__}: {exc}",),
            )

        named_target = TargetPose(**gripper_center)
        xyz_reasons = validate_xyz_values(named_target)
        if xyz_reasons:
            return _failure(
                waypoint,
                "INVALID_CARTESIAN_TARGET",
                xyz_reasons,
            )
        workspace_reasons = validate_workspace(
            named_target,
            self.kinematics_settings,
        )
        if workspace_reasons:
            return _failure(
                waypoint,
                "WORKSPACE_VIOLATION",
                workspace_reasons,
            )

        return self._accepted_motion(
            replace(waypoint, cartesian_target=named_target),
            joint_angles=joint_angles,
            pulses_us=pulses,
            gripper_center=gripper_center,
        )

    def _validate_gripper_waypoint(
        self,
        waypoint: Waypoint,
    ) -> PlannedMotion | PlanningFailure:
        pulse_key = waypoint.gripper_pulse_key
        assert pulse_key is not None
        if waypoint.cartesian_target is not None:
            xyz_reasons = validate_xyz_values(waypoint.cartesian_target)
            if xyz_reasons:
                return _failure(
                    waypoint,
                    "INVALID_CARTESIAN_TARGET",
                    xyz_reasons,
                )
            workspace_reasons = validate_workspace(
                waypoint.cartesian_target,
                self.kinematics_settings,
            )
            if workspace_reasons:
                return _failure(
                    waypoint,
                    "WORKSPACE_VIOLATION",
                    workspace_reasons,
                )

        commands = self.poses.get("gripper_commands", {})
        if pulse_key not in commands:
            return _failure(
                waypoint,
                "INVALID_GRIPPER_COMMAND",
                (f"Unknown gripper pulse key {pulse_key!r}",),
            )

        try:
            pulse_us = int(commands[pulse_key])
        except (TypeError, ValueError) as exc:
            return _failure(
                waypoint,
                "INVALID_GRIPPER_COMMAND",
                (f"{type(exc).__name__}: {exc}",),
            )

        gripper_joint = str(self.kinematics_settings["model"]["gripper_joint"])
        pulse_reasons = validate_gripper_pulse(
            gripper_joint,
            pulse_us,
            self.servo_calibration,
        )
        if pulse_reasons:
            return _failure(
                waypoint,
                "SERVO_PULSE_LIMIT_VIOLATION",
                pulse_reasons,
            )

        return self._accepted_motion(
            waypoint,
            joint_angles=None,
            pulses_us={gripper_joint: pulse_us},
            gripper_center=(
                waypoint.cartesian_target.as_dict()
                if waypoint.cartesian_target is not None
                else None
            ),
        )

    def _accepted_motion(
        self,
        waypoint: Waypoint,
        *,
        joint_angles: dict[str, float] | None,
        pulses_us: dict[str, int],
        gripper_center: dict[str, float] | None,
        ik_branch: str | None = None,
    ) -> PlannedMotion | PlanningFailure:
        hardware_reasons = validate_hardware_safe_pulses(
            pulses_us,
            self.servo_calibration,
        )
        if hardware_reasons and self.enforce_hardware_safe_limits:
            return _failure(
                waypoint,
                "HARDWARE_SAFE_LIMIT_VIOLATION",
                hardware_reasons,
            )

        warnings = tuple(
            f"Hardware preflight warning: {reason}"
            for reason in hardware_reasons
        )
        accepted = replace(
            waypoint,
            joint_angles_deg=(
                dict(joint_angles)
                if joint_angles is not None
                else None
            ),
            pulses_us=dict(pulses_us),
            ik_branch=ik_branch,
            warnings=warnings,
            validation_status=ValidationStatus.VALID,
            rejection_reasons=(),
        )
        command = MotionCommand(
            name=waypoint.command_name,
            pulses_us=dict(pulses_us),
            joint_angles_deg=(
                dict(joint_angles)
                if joint_angles is not None
                else None
            ),
            gripper_center_mm=(
                dict(gripper_center)
                if gripper_center is not None
                else None
            ),
        )
        return PlannedMotion(waypoint=accepted, command=command)

    def _joint_angles_from_ik(
        self,
        result: dict[str, Any],
    ) -> dict[str, float]:
        angles_by_label = result["angles_deg"]
        joint_angles: dict[str, float] = {}
        for joint_name, joint in self.servo_calibration["joints"].items():
            role = joint.get("kinematic_role")
            result_key = _IK_RESULT_KEYS_BY_ROLE.get(str(role))
            if result_key is not None:
                joint_angles[joint_name] = float(angles_by_label[result_key])
        return joint_angles
