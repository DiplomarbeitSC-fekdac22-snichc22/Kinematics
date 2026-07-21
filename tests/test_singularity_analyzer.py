from kinematics.singularity_analyzer import analyze_configuration


REGULAR_ANGLES = {
    "J1_base": 15.0,
    "J2_shoulder": -100.0,
    "J3_elbow": 50.0,
    "J4_wrist": -32.0,
}


def test_regular_configuration_is_full_rank() -> None:
    result = analyze_configuration(REGULAR_ANGLES)
    assert result.metrics.rank == 4
    assert result.geometric_status == "regular"
    assert not result.base_axis_singularity
    assert not result.elbow_singularity


def test_extended_elbow_is_singular() -> None:
    angles = dict(REGULAR_ANGLES)
    angles["J3_elbow"] = 180.0
    result = analyze_configuration(angles)

    assert result.elbow_singularity
    assert result.geometric_status == "singular"
    assert result.metrics.rank < 4


def test_folded_elbow_is_singular() -> None:
    angles = dict(REGULAR_ANGLES)
    angles["J3_elbow"] = 0.0
    result = analyze_configuration(angles)

    assert result.elbow_singularity
    assert result.geometric_status == "singular"
    assert result.metrics.rank < 4


def test_condition_worsens_when_approaching_extended_elbow() -> None:
    farther = dict(REGULAR_ANGLES)
    nearer = dict(REGULAR_ANGLES)
    farther["J3_elbow"] = 150.0
    nearer["J3_elbow"] = 179.0

    farther_result = analyze_configuration(farther)
    nearer_result = analyze_configuration(nearer)

    assert (
        nearer_result.metrics.inverse_condition_number
        < farther_result.metrics.inverse_condition_number
    )
