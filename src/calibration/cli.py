import argparse
import sys
from pathlib import Path
from typing import Sequence

from calibration.fitting import (
    DEFAULT_MINIMUM_PAIRS,
    CalibrationDataError,
    fit_servo_calibrations,
)
from calibration.io import (
    load_measurements_csv,
    load_servo_calibration,
    write_calibration_proposal,
    write_calibration_report,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SERVO_CONFIG = REPOSITORY_ROOT / "configs" / "servo_calibration.toml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit servo pulse/angle calibration from multiple physical "
            "measurements per joint."
        )
    )
    parser.add_argument(
        "measurements_csv",
        type=Path,
        help="CSV containing joint_name, angle_deg, and pulse_us",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_SERVO_CONFIG,
        help="Existing servo_calibration.toml used for joint names and zeros",
    )
    parser.add_argument(
        "--minimum-pairs",
        type=int,
        default=DEFAULT_MINIMUM_PAIRS,
        help="Minimum measured pairs and distinct angles per joint (default: 3)",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=Path("servo_calibration_fit.json"),
        help="Detailed JSON fit report",
    )
    parser.add_argument(
        "--proposal-toml",
        type=Path,
        default=Path("servo_calibration_proposal.toml"),
        help="Review-only TOML snippets with suggested fitted values",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        measurements = load_measurements_csv(args.measurements_csv)
        servo_calibration = load_servo_calibration(args.config)
        fits = fit_servo_calibrations(
            measurements,
            servo_calibration,
            minimum_pairs=args.minimum_pairs,
        )
        write_calibration_report(fits, args.report_json)
        write_calibration_proposal(fits, args.proposal_toml)
    except (CalibrationDataError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"Fitted {len(fits)} joint calibration(s):")
    for joint_name, fit in fits.items():
        print(
            f"  {joint_name}: direction={fit.direction:+d}, "
            f"us_per_degree={fit.us_per_degree:.6f}, "
            f"pulse_center_us={fit.pulse_center_us:.3f}, "
            f"RMSE={fit.rmse_us:.3f} us, R^2={fit.r_squared:.6f}"
        )
    print(f"JSON report:   {args.report_json}")
    print(f"TOML proposal: {args.proposal_toml}")
    print("Review the proposal manually; the live configuration was not changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
