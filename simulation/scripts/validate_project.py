#!/usr/bin/env python3
"""Static checks for the Webots project that do not require Webots itself."""

from __future__ import annotations

import re
import sys
import tomllib
from math import radians
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SIMULATION = ROOT / "simulation"
WORLD = SIMULATION / "worlds" / "robot_arm_pick_and_place.wbt"
ROBOT_PROTO = SIMULATION / "protos" / "RobotArm.proto"
WORKCELL_PROTO = SIMULATION / "protos" / "RobotWorkcell.proto"
OBJECT_PROTO = SIMULATION / "protos" / "PickObject.proto"
CONTROLLER = (
    SIMULATION
    / "controllers"
    / "kinematics_webots"
    / "kinematics_webots.py"
)


def _without_comments_and_strings(text: str) -> str:
    text = re.sub(r"#.*", "", text)
    return re.sub(r'"(?:\\.|[^"\\])*"', '""', text)


def _assert_balanced(path: Path) -> None:
    text = _without_comments_and_strings(path.read_text(encoding="utf-8"))
    pairs = {"}": "{", "]": "["}
    stack: list[str] = []

    for character in text:
        if character in "{[":
            stack.append(character)
        elif character in pairs:
            if not stack or stack.pop() != pairs[character]:
                raise AssertionError(f"Unbalanced {character!r} in {path}")

    if stack:
        raise AssertionError(f"Unclosed delimiter(s) {stack!r} in {path}")


def _load_toml(name: str) -> dict[str, object]:
    with (ROOT / "configs" / name).open("rb") as handle:
        return tomllib.load(handle)


def main() -> int:
    required = (
        WORLD,
        ROBOT_PROTO,
        WORKCELL_PROTO,
        OBJECT_PROTO,
        CONTROLLER,
    )
    for path in required:
        if not path.is_file():
            raise AssertionError(f"Missing Webots project file: {path}")

    for path in (WORLD, ROBOT_PROTO, WORKCELL_PROTO, OBJECT_PROTO):
        text = path.read_text(encoding="utf-8")
        if not text.startswith("#VRML_SIM R2025a utf8"):
            raise AssertionError(f"Wrong Webots header in {path}")
        _assert_balanced(path)

    world_text = WORLD.read_text(encoding="utf-8")
    if "gravity 9.81" not in world_text:
        raise AssertionError(
            "WorldInfo.gravity must be the R2025a scalar value 9.81"
        )
    if re.search(r"\bgravity\s+[-+0-9.eE]+\s+[-+0-9.eE]+", world_text):
        raise AssertionError(
            "WorldInfo.gravity is scalar in R2025a, not an SFVec3f"
        )
    if "attenuation 0 0 1" not in world_text:
        raise AssertionError(
            "PointLight must use quadratic attenuation to avoid the Webots warning"
        )

    for target in re.findall(r'EXTERNPROTO\s+"([^"]+)"', world_text):
        resolved = (WORLD.parent / target).resolve()
        if not resolved.is_file():
            raise AssertionError(f"Broken EXTERNPROTO target: {target}")

    simulation = _load_toml("webots_simulation.toml")
    geometry = _load_toml("robot_geometry.toml")
    settings = _load_toml("kinematics_settings.toml")
    poses = _load_toml("poses.toml")

    model = simulation["model"]
    links = geometry["link_lengths_mm"]
    assert model["link_1_mm"] == links["L1_shoulder_to_elbow"]
    assert model["link_2_mm"] == links["L2_elbow_to_wrist"]
    assert model["tool_length_mm"] == links["Lg_selected"]
    assert (
        simulation["coordinate_mapping"]["top_reference_height_mm"]
        == settings["input_coordinates"]["max_height_mm"]
    )
    top_reference_mm = float(
        simulation["coordinate_mapping"]["top_reference_height_mm"]
    )
    shoulder_below_roof_mm = float(
        model["shoulder_distance_below_roof_mm"]
    )
    expected_shoulder_height_mm = (
        top_reference_mm - shoulder_below_roof_mm
    )
    assert model["shoulder_height_from_floor_mm"] == expected_shoulder_height_mm

    ready = poses["poses"]["ready"]
    expected_initial_positions = {
        "J1_base": 0.0,
        "J2_shoulder": float(ready["J2_shoulder"]),
        "J3_elbow": 180.0 - float(ready["J3_elbow"]),
        "J4_wrist": float(ready["J4_wrist"]),
    }
    robot_text = ROBOT_PROTO.read_text(encoding="utf-8")
    expected_robot_z = expected_shoulder_height_mm / 1000.0
    if f"translation   0 0 {expected_robot_z:.5f}" not in robot_text:
        raise AssertionError(
            "Robot shoulder origin is not synchronized with the roof mount: "
            f"expected world Z={expected_robot_z:.5f} m"
        )
    for joint, value in expected_initial_positions.items():
        config = simulation["joints"][joint]
        device = config["motor_device"]
        sensor = config["sensor_device"]
        if f'name "{device}"' not in robot_text:
            raise AssertionError(f"Missing Webots motor {device}")
        if f'name "{sensor}"' not in robot_text:
            raise AssertionError(f"Missing Webots sensor {sensor}")

        expected_radians = value * 3.141592653589793 / 180.0
        if f"position {expected_radians:.6f}" not in robot_text:
            raise AssertionError(
                f"Initial position for {joint} is not synchronized: "
                f"expected {expected_radians:.6f} rad"
            )

    for key in ("left_camera", "right_camera", "tof_range_finder"):
        name = simulation["devices"][key]
        if f'name "{name}"' not in robot_text:
            raise AssertionError(f"Missing configured Webots device {name}")

    cameras = simulation["cameras"]
    if robot_text.count("Camera {") != cameras["count"]:
        raise AssertionError("Webots camera count does not match configuration")
    if robot_text.count(f'width {cameras["width_px"]}') < cameras["count"]:
        raise AssertionError("Webots camera width does not match configuration")
    if robot_text.count(f'height {cameras["height_px"]}') < cameras["count"]:
        raise AssertionError("Webots camera height does not match configuration")
    camera_fov = radians(float(cameras["horizontal_field_of_view_deg"]))
    if robot_text.count(f"fieldOfView {camera_fov:.6f}") < cameras["count"]:
        raise AssertionError("Webots camera field of view does not match configuration")
    for side in ("left", "right"):
        translation = cameras[f"{side}_translation_m"]
        expected = "translation " + " ".join(
            f"{float(value):.5f}" if index == 2 else f"{float(value):.3f}"
            for index, value in enumerate(translation)
        )
        if robot_text.count(expected) < 2:
            raise AssertionError(
                f"Webots {side} camera pose does not match configuration: {expected}"
            )
    if robot_text.count("translation 0 0 0.006") < cameras["count"]:
        raise AssertionError(
            "Camera housings must sit behind, not around, the optical centres"
        )

    # R2025a requires HingeJoint.maxStop to be strictly less than pi.
    if "maxStop 3.141593" in robot_text or "maxStop 3.141592" not in robot_text:
        raise AssertionError("J3 maxStop must be represented just below pi")

    # Slider endpoint translations must include the configured initial joint
    # displacement. Omitting it makes Webots snap the jaw into a new reference
    # pose when physics starts.
    open_position = float(simulation["gripper"]["open_slider_position_m"])
    for expected in (
        f"translation 0.119 {open_position:.3f} 0",
        f"translation 0.119 {-open_position:.3f} 0",
    ):
        if expected not in robot_text:
            raise AssertionError(
                f"Gripper slider reference pose is inconsistent: {expected}"
            )

    tof = simulation["tof"]
    tof_checks = (
        f'width {tof["grid_width"]}',
        f'height {tof["grid_height"]}',
        f'minRange {float(tof["minimum_range_m"]):.3f}',
        f'maxRange {float(tof["maximum_range_m"]):.1f}',
        "fieldOfView "
        f'{radians(float(tof["webots_axis_field_of_view_deg"])):.6f}',
    )
    for check in tof_checks:
        if check not in robot_text:
            raise AssertionError(f"Webots ToF field is not synchronized: {check}")

    workcell_text = WORKCELL_PROTO.read_text(encoding="utf-8")
    x_max_m = (
        float(settings["workspace_bounds_robot_base_mm"]["x_max"]) / 1000.0
    )
    shelf_depth_m = float(settings["shelving_mm"]["depth"]) / 1000.0
    shelf_center_m = x_max_m + shelf_depth_m / 2.0
    shelf_floor_checks = (
        f"translation {shelf_center_m:.3f} 0 0.2525",
        f"translation {shelf_center_m:.3f} 0 0.1275",
        f"size {shelf_depth_m:.3f} 0.340 0.005",
    )
    for check in shelf_floor_checks:
        if check not in workcell_text:
            raise AssertionError(
                f"Webots shelf extension is not synchronized: {check}"
            )

    print("Webots static validation: PASS")
    print(f"World: {WORLD.relative_to(ROOT)}")
    print("Geometry/configuration/device mappings are synchronized.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, KeyError, OSError, tomllib.TOMLDecodeError) as error:
        print(f"Webots static validation: FAIL - {error}", file=sys.stderr)
        raise SystemExit(1)
