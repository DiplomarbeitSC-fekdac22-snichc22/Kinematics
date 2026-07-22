import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from math import dist, isfinite, sqrt
from pathlib import Path
from statistics import fmean, median, pstdev
from typing import Any, Iterable, Mapping


_REQUESTED_FIELDS = (
    "requested_x_mm",
    "requested_y_mm",
    "requested_z_mm",
)
_OBSERVED_FIELDS = (
    "observed_x_mm",
    "observed_y_mm",
    "observed_z_mm",
)
_REQUIRED_FIELDS = (*_REQUESTED_FIELDS, *_OBSERVED_FIELDS, "success")
_TRUE_VALUES = frozenset({"1", "true", "yes", "y"})
_FALSE_VALUES = frozenset({"0", "false", "no", "n"})

Position = tuple[float, float, float]


@dataclass(frozen=True)
class PositionObservation:
    """One requested move and its camera-observed result."""
    requested_mm: Position
    observed_mm: Position | None
    success: bool

    def __post_init__(self) -> None:
        _validate_position(self.requested_mm, "requested position")
        if self.success and self.observed_mm is None:
            raise ValueError(
                "A successful observation requires a camera-measured position"
            )
        if self.observed_mm is not None:
            _validate_position(self.observed_mm, "observed position")


@dataclass(frozen=True)
class ObservedPositionReport:
    """Aggregate physical positioning metrics."""
    total_attempts: int
    successful_attempts: int
    failed_attempts: int
    success_rate: float
    axis_bias_x_mm: float | None
    axis_bias_y_mm: float | None
    axis_bias_z_mm: float | None
    mean_euclidean_error_mm: float | None
    rmse_mm: float | None
    median_error_mm: float | None
    maximum_error_mm: float | None
    standard_deviation_mm: float | None
    repeatability_mm: float | None
    repeatability_target_count: int
    repeatability_sample_count: int

    def as_dict(self) -> dict[str, Any]:
        """Return a stable JSON-serializable representation."""
        return {
            "attempts": {
                "total": self.total_attempts,
                "successful": self.successful_attempts,
                "failed": self.failed_attempts,
            },
            "success_rate": self.success_rate,
            "axis_bias_mm": {
                "x": self.axis_bias_x_mm,
                "y": self.axis_bias_y_mm,
                "z": self.axis_bias_z_mm,
            },
            "euclidean_error_mm": {
                "mean": self.mean_euclidean_error_mm,
                "rmse": self.rmse_mm,
                "median": self.median_error_mm,
                "maximum": self.maximum_error_mm,
                "standard_deviation": self.standard_deviation_mm,
            },
            "repeatability_mm": self.repeatability_mm,
            "repeatability_target_count": self.repeatability_target_count,
            "repeatability_sample_count": self.repeatability_sample_count,
        }


def _validate_position(position: Position, label: str) -> None:
    if len(position) != 3 or not all(isfinite(value) for value in position):
        raise ValueError(f"{label} must contain three finite coordinates")


def _parse_float(
    row: Mapping[str, str | None],
    field: str,
    row_number: int,
) -> float:
    raw = row.get(field)
    if raw is None or not raw.strip():
        raise ValueError(f"CSV row {row_number}: {field} is required")
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(
            f"CSV row {row_number}: {field} must be numeric"
        ) from exc
    if not isfinite(value):
        raise ValueError(f"CSV row {row_number}: {field} must be finite")
    return value


def _parse_success(raw: str | None, row_number: int) -> bool:
    if raw is None:
        raise ValueError(f"CSV row {row_number}: success is required")
    normalized = raw.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ValueError(
        f"CSV row {row_number}: success must be one of "
        "true/false, yes/no, or 1/0"
    )


def _parse_observed_position(
    row: Mapping[str, str | None],
    row_number: int,
    success: bool,
) -> tuple[float, ...] | None:
    values = [row.get(field) for field in _OBSERVED_FIELDS]
    populated = [value is not None and bool(value.strip()) for value in values]

    if not any(populated):
        if success:
            raise ValueError(
                f"CSV row {row_number}: successful trial requires all "
                "observed coordinates"
            )
        return None

    if not all(populated):
        raise ValueError(
            f"CSV row {row_number}: observed coordinates must be all filled "
            "or all blank"
        )

    return tuple(
        _parse_float(row, field, row_number)
        for field in _OBSERVED_FIELDS
    )


def load_observations(path: Path | str) -> list[PositionObservation]:
    """Load observed trials from a CSV file."""
    path = Path(path)
    observations: list[PositionObservation] = []

    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        fields = set(reader.fieldnames or ())
        missing = set(_REQUIRED_FIELDS) - fields
        if missing:
            raise ValueError(
                f"CSV is missing required columns: {tuple(sorted(missing))}"
            )

        for row_number, row in enumerate(reader, start=2):
            success = _parse_success(row.get("success"), row_number)
            requested = tuple(
                _parse_float(row, field, row_number)
                for field in _REQUESTED_FIELDS
            )
            observed = _parse_observed_position(row, row_number, success)
            observations.append(
                PositionObservation(
                    requested_mm=requested,
                    observed_mm=observed,
                    success=success,
                )
            )

    if not observations:
        raise ValueError("CSV contains no observation rows")

    return observations


def _calculate_repeatability(
    successful: Iterable[PositionObservation],
) -> tuple[float | None, int, int]:
    groups: dict[Position, list[Position]] = defaultdict(list)
    for observation in successful:
        assert observation.observed_mm is not None
        groups[observation.requested_mm].append(observation.observed_mm)

    squared_distances: list[float] = []
    repeated_target_count = 0
    repeatability_sample_count = 0

    for measured_positions in groups.values():
        if len(measured_positions) < 2:
            continue

        repeated_target_count += 1
        repeatability_sample_count += len(measured_positions)
        centroid = tuple(
            fmean(position[axis] for position in measured_positions)
            for axis in range(3)
        )
        squared_distances.extend(
            dist(position, centroid) ** 2
            for position in measured_positions
        )

    if not squared_distances:
        return None, 0, 0

    return (
        sqrt(fmean(squared_distances)),
        repeated_target_count,
        repeatability_sample_count,
    )


def evaluate_observations(
    observations: Iterable[PositionObservation],
) -> ObservedPositionReport:
    """Calculate physical accuracy, repeatability, and success metrics."""
    attempts = list(observations)
    if not attempts:
        raise ValueError("At least one observation is required")

    successful = [observation for observation in attempts if observation.success]
    total_attempts = len(attempts)
    successful_attempts = len(successful)
    failed_attempts = total_attempts - successful_attempts
    success_rate = successful_attempts / total_attempts

    repeatability, repeated_targets, repeatability_samples = (
        _calculate_repeatability(successful)
    )

    if not successful:
        return ObservedPositionReport(
            total_attempts=total_attempts,
            successful_attempts=0,
            failed_attempts=failed_attempts,
            success_rate=0.0,
            axis_bias_x_mm=None,
            axis_bias_y_mm=None,
            axis_bias_z_mm=None,
            mean_euclidean_error_mm=None,
            rmse_mm=None,
            median_error_mm=None,
            maximum_error_mm=None,
            standard_deviation_mm=None,
            repeatability_mm=None,
            repeatability_target_count=0,
            repeatability_sample_count=0,
        )

    errors_by_axis: list[Position] = []
    euclidean_errors: list[float] = []
    for observation in successful:
        assert observation.observed_mm is not None
        axis_errors = tuple(
            observed - requested
            for requested, observed in zip(
                observation.requested_mm,
                observation.observed_mm,
                strict=True,
            )
        )
        errors_by_axis.append(axis_errors)
        euclidean_errors.append(dist(observation.requested_mm, observation.observed_mm))

    axis_bias = tuple(
        fmean(error[axis] for error in errors_by_axis)
        for axis in range(3)
    )

    return ObservedPositionReport(
        total_attempts=total_attempts,
        successful_attempts=successful_attempts,
        failed_attempts=failed_attempts,
        success_rate=success_rate,
        axis_bias_x_mm=axis_bias[0],
        axis_bias_y_mm=axis_bias[1],
        axis_bias_z_mm=axis_bias[2],
        mean_euclidean_error_mm=fmean(euclidean_errors),
        rmse_mm=sqrt(fmean(error**2 for error in euclidean_errors)),
        median_error_mm=median(euclidean_errors),
        maximum_error_mm=max(euclidean_errors),
        standard_deviation_mm=pstdev(euclidean_errors),
        repeatability_mm=repeatability,
        repeatability_target_count=repeated_targets,
        repeatability_sample_count=repeatability_samples,
    )


def format_report(report: ObservedPositionReport) -> str:
    """Format a report for terminal output."""
    def metric(value: float | None) -> str:
        return "N/A" if value is None else f"{value:.6f} mm"

    return "\n".join(
        (
            "Observed positioning evaluation",
            f"Total attempts:           {report.total_attempts}",
            f"Successful attempts:      {report.successful_attempts}",
            f"Failed attempts:          {report.failed_attempts}",
            f"Success rate:             {report.success_rate:.2%}",
            f"Axis bias X:              {metric(report.axis_bias_x_mm)}",
            f"Axis bias Y:              {metric(report.axis_bias_y_mm)}",
            f"Axis bias Z:              {metric(report.axis_bias_z_mm)}",
            f"Mean Euclidean error:     {metric(report.mean_euclidean_error_mm)}",
            f"RMSE:                     {metric(report.rmse_mm)}",
            f"Median error:             {metric(report.median_error_mm)}",
            f"Maximum error:            {metric(report.maximum_error_mm)}",
            f"Standard deviation:       {metric(report.standard_deviation_mm)}",
            f"Repeatability:            {metric(report.repeatability_mm)}",
            "Repeatability targets:    "
            f"{report.repeatability_target_count}",
            "Repeatability samples:    "
            f"{report.repeatability_sample_count}",
        )
    )


def write_report(report: ObservedPositionReport, path: Path | str) -> None:
    """Write the aggregate report as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(report.as_dict(), file, indent=2)
        file.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare requested robot positions with camera-observed positions. "
            "Coordinates must be in the robot frame and millimetres."
        )
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        help=(
            "CSV with requested_x_mm, requested_y_mm, requested_z_mm, "
            "observed_x_mm, observed_y_mm, observed_z_mm, and success"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON report path",
    )
    args = parser.parse_args()

    report = evaluate_observations(load_observations(args.input_csv))
    print(format_report(report))
    if args.output is not None:
        write_report(report, args.output)


if __name__ == "__main__":
    main()
