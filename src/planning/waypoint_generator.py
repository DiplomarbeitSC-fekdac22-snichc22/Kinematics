from math import hypot
from pathlib import Path
from typing import Any

from config.config_loader import DEFAULT_CONFIG_DIR, load_config
from planning.models import TargetPose, Waypoint


class WaypointGenerationError(ValueError):
    """Raised when a configured Cartesian offset cannot be applied."""
    def __init__(self, waypoint: str, message: str) -> None:
        self.waypoint = waypoint
        super().__init__(message)


class WaypointGenerator:
    """Generate all geometric and action waypoints without solving IK."""
    def __init__(
        self,
        *,
        config_dir: Path | str = DEFAULT_CONFIG_DIR,
        kinematics_settings: dict[str, Any] | None = None,
        poses_config: dict[str, Any] | None = None,
    ) -> None:
        self.config_dir = Path(config_dir)
        self.kinematics_settings = kinematics_settings or load_config(
            "kinematics_settings.toml",
            self.config_dir,
        )
        self.poses_config = poses_config or load_config(
            "poses.toml",
            self.config_dir,
        )

    def generate(self, target: TargetPose) -> tuple[Waypoint, ...]:
        """Return the full unvalidated sequence in execution order."""
        pre_grasp = self.target_in_front_of_object(target)
        grasp = self.target_at_grasp_depth(target)
        lift = self.target_lifted_from_object(target)
        retract = self.target_retracted_from_shelf(target)
        deposit = self.deposit_target()

        return (
            Waypoint(
                name="ready",
                command_name="move_ready",
                named_pose="ready",
            ),
            Waypoint(
                name="pre_grasp",
                command_name="move_in_front_of_object",
                cartesian_target=pre_grasp,
            ),
            Waypoint(
                name="grasp",
                command_name="advance_towards_object",
                cartesian_target=grasp,
            ),
            Waypoint(
                name="close_gripper",
                command_name="close_gripper",
                cartesian_target=grasp,
                gripper_pulse_key="closed_pulse_us",
            ),
            Waypoint(
                name="lift",
                command_name="lift_object",
                cartesian_target=lift,
            ),
            Waypoint(
                name="retract",
                command_name="retract_from_shelf",
                cartesian_target=retract,
            ),
            Waypoint(
                name="deposit",
                command_name="move_deposit",
                cartesian_target=deposit,
            ),
            Waypoint(
                name="open_gripper",
                command_name="open_gripper",
                cartesian_target=deposit,
                gripper_pulse_key="open_pulse_us",
            ),
            Waypoint(
                name="home",
                command_name="move_home",
                named_pose="home",
            ),
        )

    def target_in_front_of_object(self, target: TargetPose) -> TargetPose:
        offsets = self.kinematics_settings["target_offsets"]
        return self._target_with_radial_retraction(
            "pre_grasp",
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

    def target_at_grasp_depth(self, target: TargetPose) -> TargetPose:
        """Align the jaw contact centre with the object centre."""
        return self._target_with_radial_retraction(
            "grasp",
            target,
            y_mm=target.y_mm,
            retraction_mm=float(
                self.kinematics_settings[
                    "target_offsets"
                ]["grasp_depth_offset_mm"]
            ),
        )

    def target_lifted_from_object(self, target: TargetPose) -> TargetPose:
        offsets = self.kinematics_settings["target_offsets"]
        return self._target_with_radial_retraction(
            "lift",
            target,
            y_mm=(
                target.y_mm
                - float(offsets["lift_after_grip_y_offset_mm"])
            ),
            retraction_mm=float(offsets["grasp_depth_offset_mm"]),
        )

    def target_retracted_from_shelf(self, target: TargetPose) -> TargetPose:
        offsets = self.kinematics_settings["target_offsets"]
        return self._target_with_radial_retraction(
            "retract",
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

    def deposit_target(self) -> TargetPose:
        drop_off = self.poses_config["cartesian_targets"]["drop_off"]
        return TargetPose(
            x_mm=float(drop_off["x_mm"]),
            y_mm=float(drop_off["y_mm"]),
            z_mm=float(drop_off["z_mm"]),
        )

    def _target_with_radial_retraction(
        self,
        waypoint: str,
        target: TargetPose,
        *,
        y_mm: float,
        retraction_mm: float,
    ) -> TargetPose:
        base_x, _, base_z = self.kinematics_settings[
            "input_coordinates"
        ]["base_rotation_axis_at_mounting_plate_mm"]

        delta_x = target.x_mm - float(base_x)
        delta_z = target.z_mm - float(base_z)
        radial_distance = hypot(delta_x, delta_z)

        if retraction_mm < 0.0:
            raise WaypointGenerationError(
                waypoint,
                "Radial retraction cannot be negative",
            )

        if radial_distance <= retraction_mm:
            raise WaypointGenerationError(
                waypoint,
                "Target is too close to apply the configured radial retraction",
            )

        scale = (radial_distance - retraction_mm) / radial_distance
        return TargetPose(
            x_mm=float(base_x) + delta_x * scale,
            y_mm=y_mm,
            z_mm=float(base_z) + delta_z * scale,
        )

