from dataclasses import dataclass


@dataclass(frozen=True)
class CartesianPosition:
    x_mm: float
    y_mm: float
    z_mm: float


@dataclass(frozen=True)
class JointAngles:
    base_deg: float
    shoulder_deg: float
    elbow_deg: float
    wrist_deg: float


@dataclass(frozen=True)
class ServoCommands:
    base_pwm: int
    shoulder_pwm: int
    elbow_pwm: int
    wrist_pwm: int