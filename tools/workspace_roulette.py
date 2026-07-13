import random
from pathlib import Path

from config.config_loader import load_config
from kinematics.inverse_kinematics import calculate_angles
from model.result_model import CartesianPosition

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"

def random_xyz(config_dir: Path) -> CartesianPosition:
    """Return a rand coordinate in the workspace box."""
    bounds = load_config("kinematics_settings.toml", config_dir)["workspace_bounds_robot_base_mm"]

    x = random.uniform(bounds["x_min"], bounds["x_max"])
    y = random.uniform(bounds["y_min"], bounds["y_max"])
    z = random.uniform(bounds["z_min"], bounds["z_max"])

    return CartesianPosition(x, y, z)

def main():
    coords = random_xyz(CONFIG_DIR)
    result = calculate_angles(coords.x_mm, coords.y_mm, coords.z_mm, CONFIG_DIR)

    print(f"Target: x={coords.x_mm:.1f}, y={coords.y_mm:.1f}, z={coords.z_mm:.1f}")
    print()

    print(f"Result: {result['reachable']}")
    print()

    angles = result["angles_deg"]
    print(f"J1: {angles['base']:.1f}°")
    print(f"J2: {angles['shoulder']:.1f}°")
    print(f"J3: {angles['elbow']:.1f}°")
    print(f"J4: {angles['wrist']:.1f}°")
    print()

    pwm = result["pwm_us"]
    print("PWM:")
    print(f"J1: {pwm['J1']} µs")
    print(f"J2: {pwm['J2']} µs")
    print(f"J3: {pwm['J3']} µs")
    print(f"J4: {pwm['J4']} µs")

if __name__ == "__main__":
    main()
