from dataclasses import dataclass, field
from enum import Enum

from kinematics.analysis_models import ConfigurationAnalysis


class ValidationStatus(str, Enum):
    """Validation state of a generated waypoint."""
    PENDING = "pending"
    VALID = "valid"
    INVALID = "invalid"


@dataclass(frozen=True)
class TargetPose:
    """Cartesian gripper-centre target in the robot-base frame."""
    x_mm: float
    y_mm: float
    z_mm: float

    def as_dict(self) -> dict[str, float]:
        return {
            "x_mm": float(self.x_mm),
            "y_mm": float(self.y_mm),
            "z_mm": float(self.z_mm),
        }


@dataclass(frozen=True)
class MotionCommand:
    """One command that can be sent to a motion-command sink."""
    name: str
    pulses_us: dict[str, int]
    gripper_center_mm: dict[str, float] | None = None
    joint_angles_deg: dict[str, float] | None = None


@dataclass(frozen=True)
class Waypoint:
    """Generated waypoint together with its planning result."""
    name: str
    command_name: str
    cartesian_target: TargetPose | None = None
    joint_angles_deg: dict[str, float] | None = None
    pulses_us: dict[str, int] = field(default_factory=dict)
    ik_branch: str | None = None
    singularity_analysis: ConfigurationAnalysis | None = None
    warnings: tuple[str, ...] = ()
    validation_status: ValidationStatus = ValidationStatus.PENDING
    rejection_reasons: tuple[str, ...] = ()
    named_pose: str | None = None
    gripper_pulse_key: str | None = None


@dataclass(frozen=True)
class PlannedMotion:
    """Validated waypoint and the exact command produced for it."""
    waypoint: Waypoint
    command: MotionCommand


@dataclass(frozen=True)
class MotionPlan:
    """Complete, immutable ordering of validated pick-and-place motions."""
    target: TargetPose
    motions: tuple[PlannedMotion, ...]
    warnings: tuple[str, ...] = ()

    @property
    def waypoints(self) -> tuple[Waypoint, ...]:
        return tuple(motion.waypoint for motion in self.motions)

    @property
    def commands(self) -> tuple[MotionCommand, ...]:
        return tuple(motion.command for motion in self.motions)

    def motion_for(self, waypoint_name: str) -> PlannedMotion:
        for motion in self.motions:
            if motion.waypoint.name == waypoint_name:
                return motion

        raise KeyError(f"Motion plan has no waypoint {waypoint_name!r}")


@dataclass(frozen=True)
class PlanningFailure:
    """Structured reason why a complete plan was rejected."""
    waypoint: str
    code: str
    message: str
    reasons: tuple[str, ...] = ()
    rejected_waypoint: Waypoint | None = None
