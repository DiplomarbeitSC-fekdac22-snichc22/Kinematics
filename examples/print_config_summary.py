import sys
from pathlib import Path

# Allows running this example directly from the package root without installation.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kinematics_config.loader import load_all_configs

cfg = load_all_configs(ROOT / "configs")
geometry = cfg["robot_geometry"]
servo = cfg["servo_calibration"]
settings = cfg["kinematics_settings"]
poses = cfg["poses"]

print("Kinematics config summary")
print("-------------------------")
print("L1 shoulder-to-elbow:", geometry["link_lengths_mm"]["L1_shoulder_to_elbow"], "mm")
print("L2 elbow-to-wrist:", geometry["link_lengths_mm"]["L2_elbow_to_wrist"], "mm")
print("Lg selected:", geometry["link_lengths_mm"]["Lg_selected"], "mm")
print("h0 selected along Y:", geometry["link_lengths_mm"]["h0_selected"], "mm")
enclosure = settings["physical_enclosure_mm"]
print(
    "Measured enclosure (H x D x W):",
    enclosure["height"],
    "x",
    enclosure["depth"],
    "x",
    enclosure["width"],
    "mm",
)
print("Maximum gripper opening:", geometry["gripper_geometry"]["max_opening_width_mm"], "mm")
drop_off = poses["cartesian_targets"]["drop_off"]
print(
    "Drop-off:",
    f"x={drop_off['x_mm']} mm, y={drop_off['y_mm']} mm from top, z={drop_off['z_mm']} mm right",
)
print("IK solution preference:", settings["ik"]["solution_preference"])
print("\nServo channels:")
for name, joint in servo["joints"].items():
    print(
        f"  {name}: channel {joint['pca9685_channel']}, direction {joint['direction']}, pulse {joint['pulse_min_us']}-{joint['pulse_max_us']} us")
