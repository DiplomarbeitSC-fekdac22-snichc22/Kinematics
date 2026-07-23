from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Any, Mapping, Sequence

from config.config_loader import DEFAULT_CONFIG_DIR, load_config
from kinematics.analysis_models import ConfigurationAnalysis
from kinematics.inverse_kinematics import ELBOW_BACK, ELBOW_FORWARD
from kinematics.singularity_analyzer import analyze_configuration


_IK_RESULT_KEYS_BY_ROLE = {
    "theta1": "base",
    "theta2": "shoulder",
    "theta3": "elbow",
    "theta4": "wrist",
}


@dataclass(frozen=True)
class RankedIKSolution:
    solution: dict[str, Any]
    joint_angles_deg: dict[str, float]
    analysis: ConfigurationAnalysis
    valid: bool
    joint_distance_deg: float
    preferred_branch: bool


def _joint_angles_from_solution(
    solution: Mapping[str, Any],
    servo_calibration: Mapping[str, Any],
) -> dict[str, float]:
    angles_by_label = solution["angles_deg"]
    joint_angles: dict[str, float] = {}

    for joint_name, joint in servo_calibration["joints"].items():
        result_key = _IK_RESULT_KEYS_BY_ROLE.get(
            str(joint.get("kinematic_role"))
        )
        if result_key is not None:
            joint_angles[joint_name] = float(angles_by_label[result_key])

    if len(joint_angles) != len(_IK_RESULT_KEYS_BY_ROLE):
        raise KeyError("Servo calibration does not define every IK joint role")

    return joint_angles


def _joint_distance_deg(
    candidate_angles_deg: Mapping[str, float],
    current_joint_angles_deg: Mapping[str, float],
) -> float:
    missing = tuple(
        joint_name
        for joint_name in candidate_angles_deg
        if joint_name not in current_joint_angles_deg
    )
    if missing:
        raise KeyError(
            f"Current joint state is missing IK joints: {missing}"
        )

    return sqrt(
        sum(
            (
                float(candidate_angle)
                - float(current_joint_angles_deg[joint_name])
            )
            ** 2
            for joint_name, candidate_angle in candidate_angles_deg.items()
        )
    )


def _rank_key(ranked: RankedIKSolution) -> tuple[Any, ...]:
    analysis = ranked.analysis
    metrics = analysis.metrics

    return (
        -int(ranked.valid),
        ranked.joint_distance_deg,
        -int(analysis.geometric_status != "singular"),
        -int(metrics.rank),
        -float(metrics.inverse_condition_number),
        -float(analysis.joint_limit_margin),
        -float(analysis.pulse_limit_margin),
        -int(ranked.preferred_branch),
    )


def rank_continuous_solutions(
    solutions: Sequence[dict[str, Any]],
    current_joint_angles_deg: Mapping[str, float],
    config_dir: Path | str = DEFAULT_CONFIG_DIR,
) -> tuple[RankedIKSolution, ...]:
    """Rank IK candidates in the configured safety and continuity order"""
    if not solutions:
        raise ValueError("At least one IK solution is required")

    config_dir = Path(config_dir)
    settings = load_config("kinematics_settings.toml", config_dir)
    servo_calibration = load_config("servo_calibration.toml", config_dir)

    preferred_branch = str(settings["ik"]["solution_preference"])
    if preferred_branch not in {ELBOW_BACK, ELBOW_FORWARD}:
        raise ValueError(
            "ik.solution_preference must be 'elbow_back' or 'elbow_forward'"
        )

    ranked_solutions: list[RankedIKSolution] = []
    for solution in solutions:
        joint_angles = _joint_angles_from_solution(
            solution,
            servo_calibration,
        )
        analysis = analyze_configuration(
            joint_angles,
            config_dir,
            elbow_relative_sign=float(solution["elbow_relative_sign"]),
        )
        valid = (
            bool(solution.get("reachable", False))
            and analysis.constraint_status != "invalid"
        )
        ranked_solutions.append(
            RankedIKSolution(
                solution=solution,
                joint_angles_deg=joint_angles,
                analysis=analysis,
                valid=valid,
                joint_distance_deg=_joint_distance_deg(
                    joint_angles,
                    current_joint_angles_deg,
                ),
                preferred_branch=(
                    str(solution.get("branch")) == preferred_branch
                ),
            )
        )

    return tuple(sorted(ranked_solutions, key=_rank_key))


def select_continuous_solution(
    solutions: Sequence[dict[str, Any]],
    current_joint_angles_deg: Mapping[str, float],
    config_dir: Path | str = DEFAULT_CONFIG_DIR,
) -> RankedIKSolution:
    """Return the highest-ranked IK solution for the current joint state"""
    return rank_continuous_solutions(
        solutions,
        current_joint_angles_deg,
        config_dir,
    )[0]
