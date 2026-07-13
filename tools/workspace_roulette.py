import random
import sys
from dataclasses import dataclass
from pathlib import Path

from config.config_loader import load_config
from kinematics.inverse_kinematics import calculate_angles
from model.result_model import CartesianPosition

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"


@dataclass(frozen=True)
class RouletteEntry:
    coords: CartesianPosition
    result: dict


def clear_screen() -> None:
    print("\033[2J\033[H", end="", flush=True)


def random_xyz(config_dir: Path) -> CartesianPosition:
    """Return a random coordinate in the workspace box."""
    bounds = load_config("kinematics_settings.toml", config_dir)["workspace_bounds_robot_base_mm"]

    x = random.uniform(bounds["x_min"], bounds["x_max"])
    y = random.uniform(bounds["y_min"], bounds["y_max"])
    z = random.uniform(bounds["z_min"], bounds["z_max"])

    return CartesianPosition(x, y, z)


def read_key() -> str:
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        value = input("[ENTER]=new target, type 'esc' to exit: ").strip().lower()
        if value == "":
            return "enter"
        if value in {"esc", "exit", "quit", "q"}:
            return "esc"
        return value[:1]

    try:
        import termios
        import tty
    except ImportError:
        value = input("[ENTER]=new target, type 'esc' to exit: ").strip().lower()
        if value == "":
            return "enter"
        if value in {"esc", "exit", "quit", "q"}:
            return "esc"
        return value[:1]

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        key = sys.stdin.read(1)
        if key == "\x1b":
            return "esc"
        if key in ("\r", "\n"):
            return "enter"
        return key
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def display_result(coords: CartesianPosition, result: dict) -> None:
    print(f"Target: x={coords.x_mm:.1f}, y={coords.y_mm:.1f}, z={coords.z_mm:.1f}")
    print(f"Result: {'REACHABLE' if result['reachable'] else 'UNREACHABLE'}")
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


def print_summary(history: list[RouletteEntry]) -> None:
    reachable = sum(1 for entry in history if entry.result["reachable"])
    rejected = len(history) - reachable
    success_rate = (reachable / len(history) * 100.0) if history else 0.0

    print(f"Reachable targets:  {reachable:>4}")
    print(f"Rejected targets:   {rejected:>4}")
    print(f"Success rate:       {success_rate:>5.1f}%")


def main() -> None:
    history: list[RouletteEntry] = []

    while True:
        clear_screen()

        coords = random_xyz(CONFIG_DIR)
        result = calculate_angles(coords.x_mm, coords.y_mm, coords.z_mm, CONFIG_DIR)
        history.append(RouletteEntry(coords=coords, result=result))

        display_result(coords, result)

        print()
        print("Press ENTER for another random coordinate or ESC for a summary and exit...", flush=True)
        print()

        key = read_key()
        if key == "esc":
            clear_screen()
            print_summary(history)
            return
        if key == "enter":
            continue

        clear_screen()
        print(f"Unsupported key: {key!r}")
        print("Press ENTER to continue or ESC to exit.")

        while True:
            key = read_key()
            if key == "esc":
                clear_screen()
                print_summary(history)
                return
            if key == "enter":
                break


if __name__ == "__main__":
    main()
