import argparse
import csv
from itertools import product
from math import sqrt
from pathlib import Path

import numpy as np

from config.config_loader import DEFAULT_CONFIG_DIR, load_config
from kinematics.forward_kinematics import calculate_gripper_center
from kinematics.inverse_kinematics import calculate_angles


def grid_values(minimum: float, maximum: float, step: float) -> list[float]:
    values = list(np.arange(minimum, maximum + step / 2, step, dtype=float))

    if values[-1] > maximum:
        values[-1] = maximum
    elif values[-1] < maximum:
        values.append(maximum)

    return values


def evaluate_accuracy(
    step_mm: float,
    config_dir: Path = DEFAULT_CONFIG_DIR,
) -> list[dict[str, float]]:
    settings = load_config("kinematics_settings.toml", config_dir)
    bounds = settings["workspace_bounds_robot_base_mm"]
    shelf = settings["shelving_mm"]

    x_max = float(bounds["x_max"])

    if shelf["x_direction"] == "positive":
        x_max += float(shelf["depth"])

    x_values = grid_values(float(bounds["x_min"]), x_max, step_mm)
    y_values = grid_values(
        float(bounds["y_min"]),
        float(bounds["y_max"]),
        step_mm,
    )
    z_values = grid_values(
        float(bounds["z_min"]),
        float(bounds["z_max"]),
        step_mm,
    )

    results: list[dict[str, float]] = []
    rejected = 0

    for x_mm, y_mm, z_mm in product(x_values, y_values, z_values):
        inverse = calculate_angles(
            x_mm,
            y_mm,
            z_mm,
            config_dir,
        )

        if not inverse["reachable"]:
            rejected += 1
            continue

        angles = inverse["angles_deg"]

        reconstructed = calculate_gripper_center(
            {
                "J1_base": angles["base"],
                "J2_shoulder": angles["shoulder"],
                "J3_elbow": angles["elbow"],
                "J4_wrist": angles["wrist"],
            },
            config_dir,
        )

        error_x = reconstructed["x_mm"] - x_mm
        error_y = reconstructed["y_mm"] - y_mm
        error_z = reconstructed["z_mm"] - z_mm

        positional_error = sqrt(
            error_x**2
            + error_y**2
            + error_z**2
        )

        results.append(
            {
                "target_x_mm": x_mm,
                "target_y_mm": y_mm,
                "target_z_mm": z_mm,
                "reconstructed_x_mm": reconstructed["x_mm"],
                "reconstructed_y_mm": reconstructed["y_mm"],
                "reconstructed_z_mm": reconstructed["z_mm"],
                "error_x_mm": error_x,
                "error_y_mm": error_y,
                "error_z_mm": error_z,
                "positional_error_mm": positional_error,
                "J1_base_deg": angles["base"],
                "J2_shoulder_deg": angles["shoulder"],
                "J3_elbow_deg": angles["elbow"],
                "J4_wrist_deg": angles["wrist"],
            }
        )

    total = len(x_values) * len(y_values) * len(z_values)

    print(f"Total targets:     {total}")
    print(f"Evaluated targets: {len(results)}")
    print(f"Rejected targets:  {rejected}")

    if results:
        errors = np.array([result["positional_error_mm"] for result in results])

        print(f"Mean error:        {errors.mean():.9f} mm")
        print(f"RMS error:         {np.sqrt(np.mean(errors**2)):.9f} mm")
        print(f"Median error:      {np.median(errors):.9f} mm")
        print(f"95th percentile:   {np.percentile(errors, 95):.9f} mm")
        print(f"Maximum error:     {errors.max():.9f} mm")

    return results


def write_csv(results: list[dict[str, float]], path: Path) -> None:
    if not results:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys(),
        )
        writer.writeheader()
        writer.writerows(results)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate inverse/forward kinematics accuracy."
    )
    parser.add_argument(
        "--step-mm",
        type=float,
        default=20.0,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("kinematics_accuracy.csv"),
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=DEFAULT_CONFIG_DIR,
    )

    args = parser.parse_args()

    results = evaluate_accuracy(
        step_mm=args.step_mm,
        config_dir=args.config_dir,
    )

    write_csv(results, args.output)


if __name__ == "__main__":
    main()