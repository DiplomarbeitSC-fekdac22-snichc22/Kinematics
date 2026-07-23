import argparse
import math
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config.config_loader import CONFIG_FILES, DEFAULT_CONFIG_DIR
from kinematics.forward_kinematics import calculate_gripper_center
from kinematics.inverse_kinematics import calculate_angles
from kinematics.singularity_policy import singularity_policy_from_settings

REQUIRED_KINEMATIC_ROLES = {
    "theta1",
    "theta2",
    "theta3",
    "theta4",
    "gripper_command",
}

IK_RESULT_KEYS = {
    "theta1": "base",
    "theta2": "shoulder",
    "theta3": "elbow",
    "theta4": "wrist",
}

REQUIRED_SECTIONS = {
    "robot_geometry": {
        "link_lengths_mm",
        "gripper_geometry",
    },
    "servo_calibration": {
        "defaults",
        "joints",
    },
    "kinematics_settings": {
        "model",
        "input_coordinates",
        "ik",
        "fk",
        "target_offsets",
        "validation",
    },
    "pca9685": {
        "conversion",
        "channel_map",
    },
    "poses": {
        "poses",
        "cartesian_targets",
        "gripper_commands",
    },
    "physical_measurements_required": {
        "items",
    },
    "webots_simulation": {
        "coordinate_mapping",
        "joints",
        "gripper",
        "model",
    },
}


@dataclass(frozen=True)
class CheckResult:
    """One configuration-check result."""
    level: str
    message: str


class ConfigurationChecker:
    """Run syntax, structure, and cross-file configuration checks."""

    def __init__(
            self,
            config_dir: Path | str = DEFAULT_CONFIG_DIR,
            warning_margin_deg: float = 5.0,
    ) -> None:
        self.config_dir = Path(config_dir)
        self.warning_margin_deg = warning_margin_deg

        self.configs: dict[str, dict[str, Any]] = {}
        self.results: list[CheckResult] = []

    def _record(self, level: str, message: str) -> None:
        self.results.append(CheckResult(level, message))

    def ok(self, message: str) -> None:
        self._record("OK", message)

    def warn(self, message: str) -> None:
        self._record("WARN", message)

    def fail(self, message: str) -> None:
        self._record("FAIL", message)

    @property
    def failure_count(self) -> int:
        return sum(
            result.level == "FAIL"
            for result in self.results
        )

    def print_report(self) -> None:
        """Print all results and a final summary."""
        print("Robot configuration check")
        print()

        for result in self.results:
            print(f"[{result.level}] {result.message}")

        counts = {
            level: sum(
                result.level == level
                for result in self.results
            )
            for level in ("OK", "WARN", "FAIL")
        }

        print()
        print(
            "Summary: "
            f"{counts['OK']} OK, "
            f"{counts['WARN']} WARN, "
            f"{counts['FAIL']} FAIL"
        )

    def _report_error_group(
            self,
            errors: list[str],
            success_message: str,
    ) -> None:
        if errors:
            for error in errors:
                self.fail(error)
        else:
            self.ok(success_message)

    def load_configs(self) -> None:
        """Discover and parse all TOML files."""
        if not self.config_dir.is_dir():
            self.fail(
                "Configuration directory does not exist: "
                f"{self.config_dir}"
            )
            return

        discovered_paths = sorted(
            self.config_dir.glob("*.toml")
        )
        discovered_names = {
            path.name
            for path in discovered_paths
        }
        required_names = set(CONFIG_FILES)

        missing_names = sorted(
            required_names - discovered_names
        )
        extra_names = sorted(
            discovered_names - required_names
        )

        for filename in missing_names:
            self.fail(
                "Required configuration file is absent: "
                f"{filename}"
            )

        for filename in extra_names:
            self.warn(
                "Unregistered TOML file will only receive syntax "
                f"validation: {filename}"
            )

        parse_failures = 0

        for path in discovered_paths:
            try:
                with path.open("rb") as handle:
                    config = tomllib.load(handle)
            except tomllib.TOMLDecodeError as error:
                parse_failures += 1
                self.fail(
                    f"{path.name} contains invalid TOML: {error}"
                )
                continue
            except OSError as error:
                parse_failures += 1
                self.fail(
                    f"{path.name} could not be read: {error}"
                )
                continue

            self.configs[path.stem] = config

            if "schema_version" not in config:
                self.warn(
                    f"{path.name} has no schema_version"
                )

        if not missing_names and parse_failures == 0:
            self.ok(
                f"All {len(required_names)} required TOML files "
                "are present and syntactically valid"
            )

    def check_required_sections(self) -> None:
        """Check required top-level sections in known files."""
        for config_name, required_sections in (
                REQUIRED_SECTIONS.items()
        ):
            config = self.configs.get(config_name)

            if config is None:
                continue

            missing_sections = sorted(
                section
                for section in required_sections
                if section not in config
            )

            if missing_sections:
                self.fail(
                    f"{config_name}.toml is missing required "
                    f"section(s): {', '.join(missing_sections)}"
                )
            else:
                self.ok(
                    f"{config_name}.toml contains all required "
                    "top-level sections"
                )

    def check_target_offsets(self) -> None:
        """Check that every motion target offset is configured."""
        settings = self.configs["kinematics_settings"]
        offsets = settings.get("target_offsets")

        if not isinstance(offsets, dict):
            self.fail(
                "kinematics_settings.toml is missing "
                "target_offsets"
            )
            return

        required_fields = {
            "pre_grasp_y_offset_mm",
            "lift_after_grip_y_offset_mm",
            "approach_r_offset_mm",
            "grasp_depth_offset_mm",
        }
        missing_fields = sorted(
            required_fields - offsets.keys()
        )

        if missing_fields:
            self.fail(
                "kinematics_settings.toml is missing required "
                "target_offsets field(s): "
                f"{', '.join(missing_fields)}"
            )
        else:
            self.ok(
                "kinematics_settings.toml contains all required "
                "target_offsets fields"
            )

    def check_singularity_policy(self) -> None:
        """Validate planning thresholds for singular configurations."""
        settings = self.configs["kinematics_settings"]
        try:
            singularity_policy_from_settings(settings)
        except (KeyError, TypeError, ValueError) as error:
            self.fail(f"Invalid validation.singularity policy: {error}")
        else:
            self.ok(
                "kinematics_settings.toml contains a valid "
                "validation.singularity policy"
            )

    def check_servo_configuration(self) -> None:
        """Validate servo limits, channels, roles, and pulses."""
        servo = self.configs["servo_calibration"]
        pca9685 = self.configs["pca9685"]

        joints = servo["joints"]
        defaults = servo["defaults"]
        channel_map = pca9685["channel_map"]

        channel_count = int(
            pca9685["channel_count"]
        )
        resolution_counts = int(
            pca9685["resolution_counts"]
        )
        period_us = float(
            pca9685["period_us"]
        )

        electrical_min_us = float(
            defaults["pulse_electrical_min_us"]
        )
        electrical_max_us = float(
            defaults["pulse_electrical_max_us"]
        )

        required_joint_fields = {
            "pca9685_channel",
            "kinematic_role",
            "theta_zero_deg",
            "theta_home_deg",
            "theta_min_deg",
            "theta_max_deg",
            "direction",
            "pulse_center_us",
            "pulse_min_us",
            "pulse_max_us",
            "us_per_degree",
            "requires_physical_calibration",
        }

        channels: dict[int, list[str]] = {}
        roles: dict[str, list[str]] = {}

        angle_limit_errors: list[str] = []
        home_errors: list[str] = []
        pulse_errors: list[str] = []
        channel_errors: list[str] = []
        channel_map_errors: list[str] = []
        direction_errors: list[str] = []
        conversion_errors: list[str] = []

        for joint_name, joint in joints.items():
            missing_fields = sorted(
                required_joint_fields - joint.keys()
            )

            if missing_fields:
                self.fail(
                    f"{joint_name} is missing required field(s): "
                    f"{', '.join(missing_fields)}"
                )
                continue

            channel = int(
                joint["pca9685_channel"]
            )
            role = str(
                joint["kinematic_role"]
            )

            channels.setdefault(
                channel,
                [],
            ).append(joint_name)

            roles.setdefault(
                role,
                [],
            ).append(joint_name)

            if not 0 <= channel < channel_count:
                channel_errors.append(
                    f"{joint_name} uses channel {channel}, but "
                    f"valid channels are 0-{channel_count - 1}"
                )

            mapped_channel = channel_map.get(
                joint_name
            )

            if mapped_channel is None:
                channel_map_errors.append(
                    f"{joint_name} is absent from "
                    "pca9685.channel_map"
                )
            elif int(mapped_channel) != channel:
                channel_map_errors.append(
                    f"{joint_name} uses channel {channel} in "
                    "servo_calibration.toml but channel "
                    f"{mapped_channel} in pca9685.toml"
                )

            theta_min = float(
                joint["theta_min_deg"]
            )
            theta_max = float(
                joint["theta_max_deg"]
            )
            theta_home = float(
                joint["theta_home_deg"]
            )
            theta_zero = float(
                joint["theta_zero_deg"]
            )

            if theta_min > theta_max:
                angle_limit_errors.append(
                    f"{joint_name} has theta_min_deg="
                    f"{theta_min:g} greater than theta_max_deg="
                    f"{theta_max:g}"
                )
            else:
                if not theta_min <= theta_home <= theta_max:
                    home_errors.append(
                        f"{joint_name} home angle "
                        f"{theta_home:g}° is outside "
                        f"{theta_min:g}° to {theta_max:g}°"
                    )

                if not theta_min <= theta_zero <= theta_max:
                    angle_limit_errors.append(
                        f"{joint_name} zero angle "
                        f"{theta_zero:g}° is outside "
                        f"{theta_min:g}° to {theta_max:g}°"
                    )

            direction = int(
                joint["direction"]
            )

            if direction not in (-1, 1):
                direction_errors.append(
                    f"{joint_name} direction must be -1 or 1, "
                    f"not {direction}"
                )

            us_per_degree = float(
                joint["us_per_degree"]
            )

            if (
                    not math.isfinite(us_per_degree)
                    or us_per_degree <= 0
            ):
                conversion_errors.append(
                    f"{joint_name} us_per_degree must be a "
                    f"positive finite value, not {us_per_degree}"
                )

            pulse_min = float(
                joint["pulse_min_us"]
            )
            pulse_center = float(
                joint["pulse_center_us"]
            )
            pulse_max = float(
                joint["pulse_max_us"]
            )

            if pulse_min > pulse_max:
                pulse_errors.append(
                    f"{joint_name} pulse_min_us exceeds "
                    "pulse_max_us"
                )
                continue

            if not pulse_min <= pulse_center <= pulse_max:
                pulse_errors.append(
                    f"{joint_name} center pulse "
                    f"{pulse_center:g} µs is outside "
                    f"{pulse_min:g}-{pulse_max:g} µs"
                )

            if (
                    pulse_min < electrical_min_us
                    or pulse_max > electrical_max_us
            ):
                pulse_errors.append(
                    f"{joint_name} pulse range "
                    f"{pulse_min:g}-{pulse_max:g} µs is "
                    "outside the configured global electrical "
                    f"range {electrical_min_us:g}-"
                    f"{electrical_max_us:g} µs"
                )

            for label, pulse in (
                    ("minimum", pulse_min),
                    ("center", pulse_center),
                    ("maximum", pulse_max),
            ):
                count = round(
                    pulse
                    / period_us
                    * resolution_counts
                )

                if not 0 <= pulse < period_us:
                    pulse_errors.append(
                        f"{joint_name} {label} pulse "
                        f"{pulse:g} µs does not fit inside the "
                        f"{period_us:g} µs PWM period"
                    )
                elif not 0 <= count < resolution_counts:
                    pulse_errors.append(
                        f"{joint_name} {label} pulse converts "
                        f"to PCA9685 count {count}, outside "
                        f"0-{resolution_counts - 1}"
                    )

            for command_name in (
                    "pulse_open_us",
                    "pulse_closed_us",
            ):
                if command_name not in joint:
                    continue

                command_value = float(
                    joint[command_name]
                )

                if not pulse_min <= command_value <= pulse_max:
                    pulse_errors.append(
                        f"{joint_name}.{command_name}="
                        f"{command_value:g} µs is outside "
                        f"{pulse_min:g}-{pulse_max:g} µs"
                    )

            if bool(
                    joint["requires_physical_calibration"]
            ):
                self.warn(
                    "Physical calibration is still required "
                    f"for {joint_name}"
                )

        duplicate_channels = {
            channel: names
            for channel, names in channels.items()
            if len(names) > 1
        }

        if duplicate_channels:
            for channel, names in (
                    duplicate_channels.items()
            ):
                self.fail(
                    f"PCA9685 channel {channel} is assigned "
                    f"to multiple joints: {', '.join(names)}"
                )
        else:
            self.ok(
                f"All {len(joints)} joints have unique "
                "PCA9685 channels"
            )

        missing_roles = sorted(
            REQUIRED_KINEMATIC_ROLES - roles.keys()
        )
        duplicate_roles = {
            role: names
            for role, names in roles.items()
            if len(names) > 1
        }

        if missing_roles:
            self.fail(
                "Missing kinematic role(s): "
                f"{', '.join(missing_roles)}"
            )

        for role, names in duplicate_roles.items():
            self.fail(
                f"Kinematic role {role!r} is assigned more "
                f"than once: {', '.join(names)}"
            )

        if not missing_roles and not duplicate_roles:
            self.ok(
                "All required kinematic roles are assigned "
                "exactly once"
            )

        self._report_error_group(
            angle_limit_errors,
            "All joint minimum, zero, and maximum angles "
            "are ordered correctly",
        )
        self._report_error_group(
            home_errors,
            "All configured servo home angles are within "
            "joint limits",
        )
        self._report_error_group(
            pulse_errors,
            "All servo pulse ranges fit the electrical and "
            "PCA9685 limits",
        )
        self._report_error_group(
            channel_errors,
            "All PCA9685 channel numbers are valid",
        )
        self._report_error_group(
            channel_map_errors,
            "The PCA9685 channel map matches the servo "
            "calibration",
        )
        self._report_error_group(
            direction_errors,
            "All servo direction values are valid",
        )
        self._report_error_group(
            conversion_errors,
            "All servo angle-to-pulse conversion factors "
            "are valid",
        )

        servo_frequency = int(
            defaults["pwm_frequency_hz"]
        )
        pca_frequency = int(
            pca9685["frequency_hz"]
        )

        if servo_frequency != pca_frequency:
            self.fail(
                f"Servo PWM frequency is {servo_frequency} Hz, "
                "but the PCA9685 frequency is "
                f"{pca_frequency} Hz"
            )
        else:
            self.ok(
                "Servo and PCA9685 frequencies both use "
                f"{servo_frequency} Hz"
            )

    def check_named_poses(self) -> None:
        """Check named joint poses against servo limits."""

        servo = self.configs["servo_calibration"]
        pose_config = self.configs["poses"]

        joints = servo["joints"]
        poses = pose_config["poses"]

        for pose_name, pose in poses.items():
            display_name = (
                pose_name
                .replace("_", " ")
                .title()
            )

            missing_joints = [
                joint_name
                for joint_name in joints
                if joint_name not in pose
            ]

            if missing_joints:
                self.fail(
                    f"{display_name} pose is missing "
                    f"joint(s): {', '.join(missing_joints)}"
                )
                continue

            outside_limits: list[str] = []
            near_limits: list[str] = []

            for joint_name, joint in joints.items():
                try:
                    angle = float(
                        pose[joint_name]
                    )
                except (TypeError, ValueError):
                    outside_limits.append(
                        f"{joint_name} does not contain a "
                        "numeric angle"
                    )
                    continue

                theta_min = float(
                    joint["theta_min_deg"]
                )
                theta_max = float(
                    joint["theta_max_deg"]
                )

                if not theta_min <= angle <= theta_max:
                    outside_limits.append(
                        f"{joint_name}={angle:.2f}° outside "
                        f"{theta_min:.2f}° to "
                        f"{theta_max:.2f}°"
                    )
                    continue

                if (
                        joint["kinematic_role"]
                        == "gripper_command"
                ):
                    continue

                distance_to_min = angle - theta_min
                distance_to_max = theta_max - angle

                nearest_distance = min(
                    distance_to_min,
                    distance_to_max,
                )

                if (
                        nearest_distance
                        <= self.warning_margin_deg
                ):
                    if distance_to_min <= distance_to_max:
                        near_limits.append(
                            f"{joint_name} is "
                            f"{distance_to_min:.1f}° from its "
                            "minimum"
                        )
                    else:
                        near_limits.append(
                            f"{joint_name} is "
                            f"{distance_to_max:.1f}° from its "
                            "maximum"
                        )

            if outside_limits:
                self.fail(
                    f"{display_name} pose is outside joint "
                    f"limits: {'; '.join(outside_limits)}"
                )
            elif near_limits:
                self.warn(
                    f"{display_name} pose is close to a joint "
                    f"limit: {'; '.join(near_limits)}"
                )
            else:
                self.ok(
                    f"{display_name} pose is within joint "
                    "limits"
                )

            if bool(
                    pose.get(
                        "requires_measurement",
                        False,
                    )
            ):
                self.warn(
                    f"{display_name} pose still requires "
                    "physical measurement"
                )

            if bool(
                    pose.get(
                        "requires_verification",
                        False,
                    )
            ):
                self.warn(
                    f"{display_name} pose still requires "
                    "physical verification"
                )

    def check_gripper_commands(self) -> None:
        """Validate named gripper pulse commands."""
        servo = self.configs["servo_calibration"]
        pose_config = self.configs["poses"]

        joints = servo["joints"]
        commands = pose_config["gripper_commands"]

        gripper_joints = [
            (joint_name, joint)
            for joint_name, joint in joints.items()
            if (
                    joint.get("kinematic_role")
                    == "gripper_command"
            )
        ]

        if len(gripper_joints) != 1:
            self.fail(
                "Exactly one gripper joint is required to "
                "validate named gripper commands"
            )
            return

        joint_name, joint = gripper_joints[0]

        pulse_min = float(
            joint["pulse_min_us"]
        )
        pulse_max = float(
            joint["pulse_max_us"]
        )

        errors: list[str] = []

        for command_name in (
                "open_pulse_us",
                "closed_pulse_us",
                "hold_pulse_us",
        ):
            if command_name not in commands:
                errors.append(
                    f"gripper_commands.{command_name} is "
                    "missing"
                )
                continue

            pulse = float(
                commands[command_name]
            )

            if not pulse_min <= pulse <= pulse_max:
                errors.append(
                    f"gripper_commands.{command_name}="
                    f"{pulse:g} µs is outside the "
                    f"{joint_name} range "
                    f"{pulse_min:g}-{pulse_max:g} µs"
                )

        self._report_error_group(
            errors,
            "All named gripper commands are within the "
            "configured gripper pulse range",
        )

        if bool(
                commands.get(
                    "requires_measurement",
                    False,
                )
        ):
            self.warn(
                "Named gripper commands still require "
                "physical measurement"
            )

    def check_physical_measurements(self) -> None:
        """Report measurements that are still required."""
        measurements = self.configs[
            "physical_measurements_required"
        ]["items"]

        must_items = [
            item
            for item in measurements
            if item.get("priority") == "must"
        ]
        should_items = [
            item
            for item in measurements
            if item.get("priority") == "should"
        ]

        if must_items:
            variables = ", ".join(
                str(
                    item.get(
                        "variable",
                        "<unnamed>",
                    )
                )
                for item in must_items
            )

            self.warn(
                f"{len(must_items)} mandatory physical "
                "measurement groups are still listed: "
                f"{variables}"
            )
        else:
            self.ok(
                "No mandatory physical measurements remain"
            )

        if should_items:
            variables = ", ".join(
                str(
                    item.get(
                        "variable",
                        "<unnamed>",
                    )
                )
                for item in should_items
            )

            self.warn(
                f"{len(should_items)} recommended physical "
                "measurement groups remain: "
                f"{variables}"
            )
        else:
            self.ok(
                "No recommended physical measurements remain"
            )

    def check_geometry_consistency(self) -> None:
        """Validate geometry references and dimensions."""
        geometry = self.configs["robot_geometry"]
        settings = self.configs["kinematics_settings"]

        links = geometry["link_lengths_mm"]
        model = settings["model"]

        required_link_values = {
            "L1_shoulder_to_elbow",
            "L2_elbow_to_wrist",
        }

        for key in required_link_values:
            if key not in links:
                self.fail(
                    "robot_geometry.toml is missing "
                    f"link_lengths_mm.{key}"
                )
                continue

            value = float(
                links[key]
            )

            if (
                    not math.isfinite(value)
                    or value <= 0
            ):
                self.fail(
                    f"Geometry value link_lengths_mm.{key} "
                    f"must be positive, not {value:g}"
                )
            else:
                self.ok(
                    f"Geometry value link_lengths_mm.{key} "
                    "exists and is positive"
                )

        referenced_keys: list[str] = []

        if bool(
                model["use_gripper_offset"]
        ):
            referenced_keys.append(
                str(model["selected_Lg_key"])
            )

        if bool(
                model["use_h0_from_robot_geometry"]
        ):
            referenced_keys.append(
                str(model["selected_h0_key"])
            )

        for key in referenced_keys:
            if key not in links:
                self.fail(
                    "kinematics_settings.toml references "
                    "missing geometry value "
                    f"link_lengths_mm.{key}"
                )
                continue

            value = float(
                links[key]
            )

            if (
                    not math.isfinite(value)
                    or value <= 0
            ):
                self.fail(
                    f"Geometry value link_lengths_mm.{key} "
                    f"must be positive, not {value:g}"
                )
            else:
                self.ok(
                    f"Geometry reference "
                    f"link_lengths_mm.{key} exists and is "
                    "positive"
                )

        enclosure_height = float(
            settings["physical_enclosure_mm"]["height"]
        )
        coordinate_height = float(
            settings["input_coordinates"]["max_height_mm"]
        )

        if not math.isclose(
                enclosure_height,
                coordinate_height,
                abs_tol=1e-9,
        ):
            self.fail(
                "Physical enclosure height does not match "
                "the coordinate top reference: "
                f"{enclosure_height:g} mm versus "
                f"{coordinate_height:g} mm"
            )
        else:
            self.ok(
                "Physical enclosure height matches the "
                "coordinate top reference"
            )

        workspace = settings[
            "workspace_bounds_robot_base_mm"
        ]

        axis_pairs = (
            ("x_min", "x_max"),
            ("y_min", "y_max"),
            ("z_min", "z_max"),
        )

        workspace_errors: list[str] = []

        for minimum_key, maximum_key in axis_pairs:
            minimum = float(
                workspace[minimum_key]
            )
            maximum = float(
                workspace[maximum_key]
            )

            if minimum >= maximum:
                workspace_errors.append(
                    f"{minimum_key}={minimum:g} must be less "
                    f"than {maximum_key}={maximum:g}"
                )

        self._report_error_group(
            workspace_errors,
            "All configured workspace bounds are ordered "
            "correctly",
        )

    def check_webots_consistency(self) -> None:
        """Compare Webots values with source configurations."""
        geometry = self.configs["robot_geometry"]
        settings = self.configs["kinematics_settings"]
        simulation = self.configs["webots_simulation"]

        links = geometry["link_lengths_mm"]
        gripper_geometry = geometry[
            "gripper_geometry"
        ]
        simulation_model = simulation["model"]

        comparisons = {
            "Webots link 1": (
                float(
                    simulation_model["link_1_mm"]
                ),
                float(
                    links["L1_shoulder_to_elbow"]
                ),
            ),
            "Webots link 2": (
                float(
                    simulation_model["link_2_mm"]
                ),
                float(
                    links["L2_elbow_to_wrist"]
                ),
            ),
            "Webots tool length": (
                float(
                    simulation_model["tool_length_mm"]
                ),
                float(
                    links["Lg_selected"]
                ),
            ),
            "Webots maximum gripper opening": (
                float(
                    simulation_model[
                        "maximum_gripper_opening_mm"
                    ]
                ),
                float(
                    gripper_geometry[
                        "max_opening_width_mm"
                    ]
                ),
            ),
            "Webots top reference": (
                float(
                    simulation["coordinate_mapping"][
                        "top_reference_height_mm"
                    ]
                ),
                float(
                    settings["input_coordinates"][
                        "max_height_mm"
                    ]
                ),
            ),
            "Webots shoulder roof offset": (
                float(
                    simulation_model[
                        "shoulder_distance_below_roof_mm"
                    ]
                ),
                float(
                    links["h0_selected"]
                ),
            ),
        }

        for label, (
                simulation_value,
                source_value,
        ) in comparisons.items():
            if not math.isclose(
                    simulation_value,
                    source_value,
                    abs_tol=1e-6,
            ):
                self.fail(
                    f"{label} is inconsistent: simulation "
                    f"has {simulation_value:g}, source "
                    f"configuration has {source_value:g}"
                )
            else:
                self.ok(
                    f"{label} matches its source "
                    "configuration"
                )

        top_reference = float(
            simulation["coordinate_mapping"][
                "top_reference_height_mm"
            ]
        )
        shoulder_offset = float(
            simulation_model[
                "shoulder_distance_below_roof_mm"
            ]
        )
        configured_shoulder_height = float(
            simulation_model[
                "shoulder_height_from_floor_mm"
            ]
        )
        expected_shoulder_height = (
                top_reference - shoulder_offset
        )

        if not math.isclose(
                configured_shoulder_height,
                expected_shoulder_height,
                abs_tol=1e-6,
        ):
            self.fail(
                "Webots shoulder height is inconsistent: "
                f"expected {expected_shoulder_height:g} mm, "
                f"configured "
                f"{configured_shoulder_height:g} mm"
            )
        else:
            self.ok(
                "Webots shoulder height is consistent with "
                "the roof offset"
            )

        expected_arm_joints = {
            str(
                settings["model"]["base_joint"]
            ),
            *(
                str(name)
                for name in settings["model"][
                "planar_joints"
            ]
            ),
        }
        simulated_arm_joints = set(
            simulation["joints"]
        )

        missing_simulated_joints = sorted(
            expected_arm_joints
            - simulated_arm_joints
        )
        unexpected_simulated_joints = sorted(
            simulated_arm_joints
            - expected_arm_joints
        )

        if missing_simulated_joints:
            self.fail(
                "Webots configuration is missing arm "
                f"joint(s): "
                f"{', '.join(missing_simulated_joints)}"
            )
        else:
            self.ok(
                "All configured kinematic arm joints exist "
                "in the Webots configuration"
            )

        if unexpected_simulated_joints:
            self.warn(
                "Webots configuration contains additional "
                f"arm joint(s): "
                f"{', '.join(unexpected_simulated_joints)}"
            )

        expected_gripper = str(
            settings["model"]["gripper_joint"]
        )
        simulated_gripper = str(
            simulation["gripper"]["source_joint"]
        )

        if expected_gripper != simulated_gripper:
            self.fail(
                f"Webots gripper source is "
                f"{simulated_gripper}, expected "
                f"{expected_gripper}"
            )
        else:
            self.ok(
                "Webots gripper source matches the "
                "kinematic model"
            )

        open_position = float(
            simulation["gripper"][
                "open_slider_position_m"
            ]
        )
        closed_position = float(
            simulation["gripper"][
                "closed_slider_position_m"
            ]
        )
        minimum_functional_open = float(
            simulation["gripper"][
                "minimum_functional_open_position_m"
            ]
        )

        if not (
                closed_position
                < minimum_functional_open
                <= open_position
        ):
            self.fail(
                "Webots functional gripper opening must lie "
                "between the closed and nominal open "
                "positions"
            )
        else:
            self.ok(
                "Webots functional gripper opening lies "
                "between the closed and open positions"
            )

        positive_fields = {
            "gripper.max_velocity_m_s": float(
                simulation["gripper"][
                    "max_velocity_m_s"
                ]
            ),
            "gripper.max_acceleration_m_s2": float(
                simulation["gripper"][
                    "max_acceleration_m_s2"
                ]
            ),
            "gripper.max_force_n": float(
                simulation["gripper"][
                    "max_force_n"
                ]
            ),
            "timing.basic_time_step_ms": float(
                simulation["timing"][
                    "basic_time_step_ms"
                ]
            ),
            "timing.command_timeout_s": float(
                simulation["timing"][
                    "command_timeout_s"
                ]
            ),
        }

        webots_value_errors: list[str] = []

        for field_name, value in positive_fields.items():
            if (
                    not math.isfinite(value)
                    or value <= 0
            ):
                webots_value_errors.append(
                    f"{field_name} must be positive, "
                    f"not {value:g}"
                )

        self._report_error_group(
            webots_value_errors,
            "All required Webots timing and motor values "
            "are positive",
        )

    def _report_position_error(
            self,
            label: str,
            error_mm: float,
            warning_threshold_mm: float,
            failure_threshold_mm: float,
    ) -> None:
        if not math.isfinite(error_mm):
            self.fail(
                f"{label} produced a non-finite position "
                "error"
            )
        elif error_mm >= failure_threshold_mm:
            self.fail(
                f"{label} has an FK position error of "
                f"{error_mm:.2f} mm"
            )
        elif error_mm >= warning_threshold_mm:
            self.warn(
                f"{label} has an FK position error of "
                f"{error_mm:.2f} mm"
            )
        else:
            self.ok(
                f"{label} agrees within "
                f"{error_mm:.2f} mm"
            )

    def check_cartesian_targets(self) -> None:
        """Run IK and FK validation for named targets."""
        pose_config = self.configs["poses"]
        servo = self.configs["servo_calibration"]
        settings = self.configs["kinematics_settings"]

        targets = pose_config["cartesian_targets"]
        named_poses = pose_config["poses"]
        joints = servo["joints"]

        warning_threshold = float(
            settings["fk"][
                "position_error_warning_mm"
            ]
        )
        failure_threshold = float(
            settings["fk"][
                "position_error_fail_mm"
            ]
        )

        if warning_threshold < 0:
            self.fail(
                "FK warning threshold cannot be negative"
            )
            return

        if failure_threshold <= warning_threshold:
            self.fail(
                "FK failure threshold must be greater than "
                "the warning threshold"
            )
            return

        for target_name, target in targets.items():
            display_name = (
                target_name
                .replace("_", " ")
                .title()
            )

            try:
                x_mm = float(
                    target["x_mm"]
                )
                y_mm = float(
                    target["y_mm"]
                )
                z_mm = float(
                    target["z_mm"]
                )
            except (
                    KeyError,
                    TypeError,
                    ValueError,
            ) as error:
                self.fail(
                    f"{display_name} target has invalid "
                    f"coordinates: {error}"
                )
                continue

            result = calculate_angles(
                x_mm,
                y_mm,
                z_mm,
                self.config_dir,
            )

            if not result["reachable"]:
                reasons = "; ".join(
                    result["reasons"]
                )
                self.fail(
                    f"{display_name} target is unreachable: "
                    f"{reasons}"
                )
                continue

            self.ok(
                f"{display_name} target is reachable"
            )

            ik_joint_angles: dict[str, float] = {}

            for joint_name, joint in joints.items():
                role = str(
                    joint["kinematic_role"]
                )
                result_key = IK_RESULT_KEYS.get(
                    role
                )

                if result_key is None:
                    continue

                ik_joint_angles[joint_name] = float(
                    result["angles_deg"][result_key]
                )

            fk_position = calculate_gripper_center(
                ik_joint_angles,
                self.config_dir,
            )

            ik_fk_error = math.dist(
                (
                    x_mm,
                    y_mm,
                    z_mm,
                ),
                (
                    fk_position["x_mm"],
                    fk_position["y_mm"],
                    fk_position["z_mm"],
                ),
            )

            self._report_position_error(
                f"IK/FK result for {display_name}",
                ik_fk_error,
                warning_threshold,
                failure_threshold,
            )

            pose_name = target.get("pose")

            if pose_name is None:
                continue

            pose_name = str(
                pose_name
            )
            stored_pose = named_poses.get(
                pose_name
            )

            if stored_pose is None:
                self.fail(
                    f"{display_name} references absent "
                    f"named pose {pose_name!r}"
                )
                continue

            missing_pose_angles = [
                joint_name
                for joint_name, joint in joints.items()
                if (
                        joint["kinematic_role"]
                        in IK_RESULT_KEYS
                        and joint_name not in stored_pose
                )
            ]

            if missing_pose_angles:
                self.fail(
                    f"Named pose {pose_name!r} is missing "
                    f"angle(s): "
                    f"{', '.join(missing_pose_angles)}"
                )
                continue

            stored_angles = {
                joint_name: float(
                    stored_pose[joint_name]
                )
                for joint_name, joint in joints.items()
                if (
                        joint["kinematic_role"]
                        in IK_RESULT_KEYS
                )
            }

            stored_fk = calculate_gripper_center(
                stored_angles,
                self.config_dir,
            )

            stored_pose_error = math.dist(
                (
                    x_mm,
                    y_mm,
                    z_mm,
                ),
                (
                    stored_fk["x_mm"],
                    stored_fk["y_mm"],
                    stored_fk["z_mm"],
                ),
            )

            self._report_position_error(
                (
                    f"Named pose {pose_name!r} against "
                    f"{display_name}"
                ),
                stored_pose_error,
                warning_threshold,
                failure_threshold,
            )

    def run(self) -> bool:
        """Run every available configuration check."""
        self.load_configs()

        checks: tuple[
            tuple[str, Callable[[], None]],
            ...,
        ] = (
            (
                "required-section validation",
                self.check_required_sections,
            ),
            (
                "target-offset validation",
                self.check_target_offsets,
            ),
            (
                "singularity-policy validation",
                self.check_singularity_policy,
            ),
            (
                "servo validation",
                self.check_servo_configuration,
            ),
            (
                "named-pose validation",
                self.check_named_poses,
            ),
            (
                "gripper-command validation",
                self.check_gripper_commands,
            ),
            (
                "physical-measurement validation",
                self.check_physical_measurements,
            ),
            (
                "geometry consistency validation",
                self.check_geometry_consistency,
            ),
            (
                "Webots consistency validation",
                self.check_webots_consistency,
            ),
            (
                "Cartesian-target validation",
                self.check_cartesian_targets,
            ),
        )

        for label, check in checks:
            try:
                check()
            except (
                    KeyError,
                    TypeError,
                    ValueError,
                    ZeroDivisionError,
                    OSError,
                    tomllib.TOMLDecodeError,
            ) as error:
                self.fail(
                    f"{label} could not be completed: "
                    f"{error}"
                )

        return self.failure_count == 0


def main() -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate the robot TOML configuration."
        )
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=DEFAULT_CONFIG_DIR,
        help=(
            "Directory containing the TOML "
            "configuration files."
        ),
    )
    parser.add_argument(
        "--warning-margin-deg",
        type=float,
        default=5.0,
        help=(
            "Warn when a named pose lies this many "
            "degrees or less from a joint limit."
        ),
    )

    arguments = parser.parse_args()

    if arguments.warning_margin_deg < 0:
        parser.error(
            "--warning-margin-deg cannot be negative"
        )

    checker = ConfigurationChecker(
        config_dir=arguments.config_dir,
        warning_margin_deg=(
            arguments.warning_margin_deg
        ),
    )

    successful = checker.run()
    checker.print_report()

    return 0 if successful else 1


if __name__ == "__main__":
    raise SystemExit(main())
