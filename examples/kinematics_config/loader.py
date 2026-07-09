from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

CONFIG_FILES = (
    "robot_geometry.toml",
    "servo_calibration.toml",
    "kinematics_settings.toml",
    "pca9685.toml",
    "poses.toml",
    "physical_measurements_required.toml",
)


def load_config(path: Path | str) -> dict[str, Any]:
    """Load one TOML configuration file using Python's standard-library tomllib."""
    file_path = Path(path)
    with file_path.open("rb") as f:
        return tomllib.load(f)


def load_all_configs(config_dir: Path | str) -> dict[str, dict[str, Any]]:
    """Load all kinematics configuration files from a directory."""
    directory = Path(config_dir)
    configs: dict[str, dict[str, Any]] = {}
    for filename in CONFIG_FILES:
        path = directory / filename
        key = path.stem
        configs[key] = load_config(path)
    return configs
