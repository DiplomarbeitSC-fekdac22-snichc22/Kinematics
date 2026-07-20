import argparse
import math
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


class ServoConfigurationError(ValueError):
    """Raised when a servo configuration cannot be normalized safely."""


@dataclass(frozen=True)
class ServoDefinition:
    """Normalized servo properties shared by both repositories."""
    name: str
    channel: int
    pulse_min_us: float
    pulse_max_us: float
    neutral_pulse_us: float
    direction: int
    angle_min_deg: float
    angle_max_deg: float
    servo_model: str
    gripper_open_us: float | None = None
    gripper_closed_us: float | None = None


@dataclass(frozen=True)
class ServoConfiguration:
    """Normalized contents of one servo configuration file."""
    path: Path
    update_frequency_hz: float
    joints: Mapping[str, ServoDefinition]


@dataclass(frozen=True)
class FieldComparison:
    """One field comparison for one joint or global setting."""
    label: str
    matches: bool
    left: object
    right: object
    unit: str = ""


FIELD_SPECS: tuple[tuple[str, str, str], ...] = (
    ("channel", "channel", ""),
    ("direction", "direction", ""),
    ("pulse minimum", "pulse_min_us", "µs"),
    ("pulse maximum", "pulse_max_us", "µs"),
    ("neutral pulse", "neutral_pulse_us", "µs"),
    ("angle minimum", "angle_min_deg", "°"),
    ("angle maximum", "angle_max_deg", "°"),
    ("servo model", "servo_model", ""),
)

_FIELD_ALIASES: Mapping[str, tuple[str, ...]] = {
    "channel": ("pca9685_channel", "channel"),
    "pulse_min_us": ("pulse_min_us", "pulse_minimum_us"),
    "pulse_max_us": ("pulse_max_us", "pulse_maximum_us"),
    "neutral_pulse_us": (
        "pulse_center_us",
        "neutral_pulse_us",
        "pulse_neutral_us",
    ),
    "direction": ("direction", "servo_direction"),
    "angle_min_deg": ("theta_min_deg", "angle_min_deg", "minimum_angle_deg"),
    "angle_max_deg": ("theta_max_deg", "angle_max_deg", "maximum_angle_deg"),
    "servo_model": ("servo_type", "servo_model", "model"),
    "gripper_open_us": ("pulse_open_us", "gripper_open_us", "open_pulse_us"),
    "gripper_closed_us": (
        "pulse_closed_us",
        "gripper_closed_us",
        "closed_pulse_us",
    ),
}

_UPDATE_FREQUENCY_PATHS: tuple[tuple[str, ...], ...] = (
    ("defaults", "pwm_frequency_hz"),
    ("pca9685", "frequency_hz"),
    ("pca9685", "pwm_frequency_hz"),
    ("pwm_frequency_hz",),
    ("update_frequency_hz",),
)


def _lookup_alias(table: Mapping[str, Any], canonical_name: str) -> Any:
    for alias in _FIELD_ALIASES[canonical_name]:
        if alias in table:
            return table[alias]
    aliases = ", ".join(_FIELD_ALIASES[canonical_name])
    raise ServoConfigurationError(
        f"missing required field for {canonical_name!r}; expected one of: {aliases}"
    )


def _lookup_optional_alias(table: Mapping[str, Any], canonical_name: str) -> Any | None:
    for alias in _FIELD_ALIASES[canonical_name]:
        if alias in table:
            return table[alias]
    return None


def _lookup_nested(data: Mapping[str, Any], path: Sequence[str]) -> Any | None:
    current: Any = data
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _as_finite_float(value: Any, context: str) -> float:
    if isinstance(value, bool):
        raise ServoConfigurationError(f"{context} must be numeric, not boolean")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ServoConfigurationError(f"{context} must be numeric, not {value!r}") from error
    if not math.isfinite(result):
        raise ServoConfigurationError(f"{context} must be finite, not {result!r}")
    return result


def _as_int(value: Any, context: str) -> int:
    numeric = _as_finite_float(value, context)
    integer = int(numeric)
    if numeric != integer:
        raise ServoConfigurationError(f"{context} must be an integer, not {numeric!r}")
    return integer


def _joint_tables(data: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("joints", "servos"):
        value = data.get(key)
        if isinstance(value, Mapping):
            return value
    raise ServoConfigurationError("missing top-level [joints] or [servos] table")


def _is_gripper(name: str, table: Mapping[str, Any]) -> bool:
    descriptors = (
        name,
        str(table.get("function", "")),
        str(table.get("kinematic_role", "")),
        str(table.get("controller_name", "")),
    )
    return any("gripper" in descriptor.lower() for descriptor in descriptors)


def load_servo_configuration(path: str | Path) -> ServoConfiguration:
    """Load and normalize a supported TOML servo configuration."""

    source_path = Path(path)
    try:
        with source_path.open("rb") as handle:
            data = tomllib.load(handle)
    except FileNotFoundError as error:
        raise ServoConfigurationError(f"configuration file does not exist: {source_path}") from error
    except tomllib.TOMLDecodeError as error:
        raise ServoConfigurationError(f"invalid TOML in {source_path}: {error}") from error
    except OSError as error:
        raise ServoConfigurationError(f"could not read {source_path}: {error}") from error

    update_frequency: Any | None = None
    for candidate_path in _UPDATE_FREQUENCY_PATHS:
        update_frequency = _lookup_nested(data, candidate_path)
        if update_frequency is not None:
            break
    if update_frequency is None:
        expected = ", ".join(".".join(path) for path in _UPDATE_FREQUENCY_PATHS)
        raise ServoConfigurationError(
            f"{source_path}: missing update/PWM frequency; expected one of: {expected}"
        )
    update_frequency_hz = _as_finite_float(
        update_frequency,
        f"{source_path}: update frequency",
    )
    if update_frequency_hz <= 0:
        raise ServoConfigurationError(
            f"{source_path}: update frequency must be greater than zero"
        )

    normalized: dict[str, ServoDefinition] = {}
    channels: dict[int, str] = {}

    for joint_name, raw_joint in _joint_tables(data).items():
        if not isinstance(joint_name, str) or not joint_name:
            raise ServoConfigurationError(f"{source_path}: joint names must be non-empty strings")
        if not isinstance(raw_joint, Mapping):
            raise ServoConfigurationError(
                f"{source_path}: joint {joint_name!r} must be a TOML table"
            )

        context = f"{source_path}: {joint_name}"
        channel = _as_int(_lookup_alias(raw_joint, "channel"), f"{context} channel")
        if not 0 <= channel <= 15:
            raise ServoConfigurationError(f"{context} channel must be between 0 and 15")
        if channel in channels:
            raise ServoConfigurationError(
                f"{context} reuses PCA9685 channel {channel} already assigned to {channels[channel]}"
            )
        channels[channel] = joint_name

        direction = _as_int(
            _lookup_alias(raw_joint, "direction"),
            f"{context} direction",
        )
        if direction not in (-1, 1):
            raise ServoConfigurationError(f"{context} direction must be -1 or 1")

        pulse_min = _as_finite_float(
            _lookup_alias(raw_joint, "pulse_min_us"),
            f"{context} pulse minimum",
        )
        pulse_max = _as_finite_float(
            _lookup_alias(raw_joint, "pulse_max_us"),
            f"{context} pulse maximum",
        )
        neutral = _as_finite_float(
            _lookup_alias(raw_joint, "neutral_pulse_us"),
            f"{context} neutral pulse",
        )
        if pulse_min > pulse_max:
            raise ServoConfigurationError(f"{context} pulse minimum exceeds pulse maximum")
        if not pulse_min <= neutral <= pulse_max:
            raise ServoConfigurationError(
                f"{context} neutral pulse must lie inside the pulse range"
            )

        angle_min = _as_finite_float(
            _lookup_alias(raw_joint, "angle_min_deg"),
            f"{context} angle minimum",
        )
        angle_max = _as_finite_float(
            _lookup_alias(raw_joint, "angle_max_deg"),
            f"{context} angle maximum",
        )
        if angle_min > angle_max:
            raise ServoConfigurationError(f"{context} angle minimum exceeds angle maximum")

        model = str(_lookup_alias(raw_joint, "servo_model")).strip()
        if not model:
            raise ServoConfigurationError(f"{context} servo model must not be empty")

        open_raw = _lookup_optional_alias(raw_joint, "gripper_open_us")
        closed_raw = _lookup_optional_alias(raw_joint, "gripper_closed_us")
        if _is_gripper(joint_name, raw_joint):
            if open_raw is None or closed_raw is None:
                raise ServoConfigurationError(
                    f"{context} must define both gripper open and closed pulse values"
                )
        elif (open_raw is None) != (closed_raw is None):
            raise ServoConfigurationError(
                f"{context} must define both gripper open and closed pulse values or neither"
            )

        gripper_open = (
            _as_finite_float(open_raw, f"{context} gripper open pulse")
            if open_raw is not None
            else None
        )
        gripper_closed = (
            _as_finite_float(closed_raw, f"{context} gripper closed pulse")
            if closed_raw is not None
            else None
        )
        for label, pulse in (("open", gripper_open), ("closed", gripper_closed)):
            if pulse is not None and not pulse_min <= pulse <= pulse_max:
                raise ServoConfigurationError(
                    f"{context} gripper {label} pulse must lie inside the pulse range"
                )

        normalized[joint_name] = ServoDefinition(
            name=joint_name,
            channel=channel,
            pulse_min_us=pulse_min,
            pulse_max_us=pulse_max,
            neutral_pulse_us=neutral,
            direction=direction,
            angle_min_deg=angle_min,
            angle_max_deg=angle_max,
            servo_model=model,
            gripper_open_us=gripper_open,
            gripper_closed_us=gripper_closed,
        )

    if not normalized:
        raise ServoConfigurationError(f"{source_path}: no servo joints are configured")

    return ServoConfiguration(
        path=source_path,
        update_frequency_hz=update_frequency_hz,
        joints=normalized,
    )


def _values_match(left: object, right: object, tolerance: float) -> bool:
    if isinstance(left, (int, float)) and not isinstance(left, bool):
        if isinstance(right, (int, float)) and not isinstance(right, bool):
            return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
    return left == right


def compare_joint(
    left: ServoDefinition,
    right: ServoDefinition,
    tolerance: float = 1e-6,
) -> list[FieldComparison]:
    """Return all comparisons for a pair of normalized joints."""

    comparisons = [
        FieldComparison(
            label=label,
            matches=_values_match(getattr(left, attribute), getattr(right, attribute), tolerance),
            left=getattr(left, attribute),
            right=getattr(right, attribute),
            unit=unit,
        )
        for label, attribute, unit in FIELD_SPECS
    ]

    if left.gripper_open_us is not None or right.gripper_open_us is not None:
        left_range = (left.gripper_open_us, left.gripper_closed_us)
        right_range = (right.gripper_open_us, right.gripper_closed_us)
        matches = all(
            left_value is not None
            and right_value is not None
            and _values_match(left_value, right_value, tolerance)
            for left_value, right_value in zip(left_range, right_range, strict=True)
        )
        comparisons.append(
            FieldComparison(
                label="gripper range",
                matches=matches,
                left=left_range,
                right=right_range,
                unit="µs",
            )
        )

    return comparisons


def _format_number(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _format_value(value: object, unit: str = "") -> str:
    if isinstance(value, tuple):
        rendered = "-".join(
            "missing" if item is None else _format_value(item)
            for item in value
        )
    elif isinstance(value, float):
        rendered = _format_number(value)
    else:
        rendered = str(value)
    separator = "" if unit == "°" else " "
    return f"{rendered}{separator}{unit}".rstrip()


def _print_comparison(
    comparison: FieldComparison,
    left_label: str,
    right_label: str,
) -> None:
    status = "OK" if comparison.matches else "MISMATCH"
    print(f"  {comparison.label + ':':<19}{status}")
    if not comparison.matches:
        print(f"    {left_label + ':':<18}{_format_value(comparison.left, comparison.unit)}")
        print(f"    {right_label + ':':<18}{_format_value(comparison.right, comparison.unit)}")


def compatibility_report(
    left: ServoConfiguration,
    right: ServoConfiguration,
    *,
    left_label: str = "Kinematics",
    right_label: str = "Robot-Controller",
    tolerance: float = 1e-6,
) -> int:
    """Print a compatibility report and return the mismatch count."""

    mismatch_count = 0

    print("Servo configuration compatibility")
    print(f"{left_label}: {left.path}")
    print(f"{right_label}: {right.path}")
    print()

    frequency_comparison = FieldComparison(
        label="update frequency",
        matches=_values_match(
            left.update_frequency_hz,
            right.update_frequency_hz,
            tolerance,
        ),
        left=left.update_frequency_hz,
        right=right.update_frequency_hz,
        unit="Hz",
    )
    print("Global")
    _print_comparison(frequency_comparison, left_label, right_label)
    if not frequency_comparison.matches:
        mismatch_count += 1
    print()

    all_names = sorted(set(left.joints) | set(right.joints))
    for joint_name in all_names:
        print(joint_name)
        left_joint = left.joints.get(joint_name)
        right_joint = right.joints.get(joint_name)

        if left_joint is None:
            mismatch_count += 1
            print(f"  joint name:         MISSING FROM {left_label}")
            print()
            continue
        if right_joint is None:
            mismatch_count += 1
            print(f"  joint name:         MISSING FROM {right_label}")
            print()
            continue

        print("  joint name:         OK")
        for comparison in compare_joint(left_joint, right_joint, tolerance):
            _print_comparison(comparison, left_label, right_label)
            if not comparison.matches:
                mismatch_count += 1
        print()

    if mismatch_count:
        print(f"Result: INCOMPATIBLE ({mismatch_count} mismatch(es))")
    else:
        print("Result: COMPATIBLE")

    return mismatch_count


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare PCA9685 channel assignments and servo calibration values "
            "between the Kinematics and Robot-Controller TOML files."
        )
    )
    parser.add_argument("kinematics_config", type=Path)
    parser.add_argument("robot_controller_config", type=Path)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="Absolute tolerance for numeric comparisons (default: 1e-6).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    arguments = parser.parse_args(argv)

    if arguments.tolerance < 0 or not math.isfinite(arguments.tolerance):
        parser.error("--tolerance must be a finite non-negative number")

    try:
        left = load_servo_configuration(arguments.kinematics_config)
        right = load_servo_configuration(arguments.robot_controller_config)
    except ServoConfigurationError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    mismatches = compatibility_report(
        left,
        right,
        tolerance=arguments.tolerance,
    )
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
