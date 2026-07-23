from kinematics.singularity_policy import SingularityPolicy


PERMISSIVE_SINGULARITY_POLICY = SingularityPolicy(
    reject_singular=False,
    reject_near_singular=False,
    minimum_inverse_condition_number=0.0,
    minimum_joint_limit_margin=0.0,
    minimum_pulse_limit_margin=0.0,
)
