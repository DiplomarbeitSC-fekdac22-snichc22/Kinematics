import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_DIR = ROOT / "configs"

CONFIG_FILES = (
    "robot_geometry.toml",
    "servo_calibration.toml",
    "kinematics_settings.toml",
    "pca9685.toml",
    "poses.toml",
    "physical_measurements_required.toml",
)


def load_config(filename: str, config_dir: Path | str = DEFAULT_CONFIG_DIR) -> dict[str, Any]:
    """Load one TOML config file from the config directory."""
    path = Path(config_dir) / filename
    with path.open("rb") as file:
        return tomllib.load(file)


def load_configs(config_dir: Path | str = DEFAULT_CONFIG_DIR) -> dict[str, dict[str, Any]]:
    """Load all known TOML config files, keyed by filename without .toml."""
    return {Path(filename).stem: load_config(filename, config_dir) for filename in CONFIG_FILES}
