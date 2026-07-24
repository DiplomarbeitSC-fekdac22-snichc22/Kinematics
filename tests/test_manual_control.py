import main as main_module
from main import main


def _unexpected_input(_: str) -> str:
    raise AssertionError("The command unexpectedly requested user input")


def test_check_reports_cartesian_hardware_interlock(capsys) -> None:
    exit_code = main(["check"], input_fn=_unexpected_input)

    output = capsys.readouterr()
    assert exit_code == 2
    assert "Cartesian hardware motion: BLOCKED" in output.out
    assert "J1_base still requires physical calibration" in output.out


def test_recorded_pose_defaults_to_dry_run(capsys) -> None:
    exit_code = main(
        ["pose", "ready"],
        input_fn=_unexpected_input,
    )

    output = capsys.readouterr()
    assert exit_code == 0
    assert "Command: move_ready" in output.out
    assert "J2_shoulder=1865 us" in output.out
    assert "DRY RUN" in output.out


def test_manual_coordinate_move_can_be_planned_without_backend(
    capsys,
) -> None:
    exit_code = main(
        ["move", "230", "180", "60", "--from-pose", "home"],
        input_fn=_unexpected_input,
    )

    output = capsys.readouterr()
    assert exit_code == 0
    assert "Command: move_to_coordinates" in output.out
    assert "x=230.0 mm, y=180.0 mm, z=60.0 mm" in output.out
    assert "DRY RUN ONLY" in output.out


def test_real_coordinate_move_is_blocked_before_hardware_or_confirmation(
    capsys,
) -> None:
    exit_code = main(
        [
            "move",
            "230",
            "180",
            "60",
            "--hardware",
            "--from-pose",
            "home",
        ],
        input_fn=_unexpected_input,
    )

    output = capsys.readouterr()
    assert exit_code == 2
    assert "Cartesian hardware motion is blocked" in output.err


def test_recorded_pose_hardware_session_confirms_returns_and_closes(
    monkeypatch,
) -> None:
    events: list[object] = []

    class FakePca9685MotionSink:
        requires_hardware_safe_prevalidation = True

        def __init__(self, *, config_dir) -> None:
            events.append(("open", config_dir))

        def close(self) -> None:
            events.append("close")

    class FakeMotionExecutor:
        def __init__(
            self,
            sink,
            *,
            initial_pulses_us,
            config_dir,
        ) -> None:
            events.append(("initial", dict(initial_pulses_us)))

        def send(self, command) -> None:
            events.append(("send", command.name))

    replies = iter(("MOVE", "", ""))
    monkeypatch.setattr(
        main_module,
        "Pca9685MotionSink",
        FakePca9685MotionSink,
    )
    monkeypatch.setattr(
        main_module,
        "MotionExecutor",
        FakeMotionExecutor,
    )

    exit_code = main(
        [
            "pose",
            "ready",
            "--hardware",
            "--from-pose",
            "home",
        ],
        input_fn=lambda _: next(replies),
    )

    assert exit_code == 0
    assert ("send", "hold_home") in events
    assert ("send", "move_ready") in events
    assert events.count(("send", "hold_home")) == 2
    assert events[-1] == "close"


def test_control_defaults_to_home_and_holds_each_exact_command(
    monkeypatch,
) -> None:
    events: list[object] = []

    class FakePca9685MotionSink:
        def __init__(self, *, config_dir) -> None:
            events.append(("open", config_dir))

        def send(self, command) -> None:
            events.append(
                ("send", command.name, dict(command.pulses_us))
            )

        def close(self) -> None:
            events.append("close")

    replies = iter(
        (
            "MOVE",
            "pose ready",
            "pwm 1000 1800 2200 1200 800",
            "status",
            "home",
            "release",
        )
    )
    monkeypatch.setattr(
        main_module,
        "Pca9685MotionSink",
        FakePca9685MotionSink,
    )

    exit_code = main(
        ["control", "--hardware"],
        input_fn=lambda _: next(replies),
    )

    assert exit_code == 0
    sends = [event for event in events if event[0] == "send"]
    assert sends[0][1] == "hold_pose_home"
    assert sends[0][2] == {
        "J1_base": 977,
        "J2_shoulder": 1743,
        "J3_elbow": 2134,
        "J4_wrist": 1172,
        "J5_gripper": 732,
    }
    assert sends[1][1] == "hold_pose_ready"
    assert sends[2] == (
        "send",
        "hold_direct_pwm",
        {
            "J1_base": 1000,
            "J2_shoulder": 1800,
            "J3_elbow": 2200,
            "J4_wrist": 1200,
            "J5_gripper": 800,
        },
    )
    assert sends[3][1] == "hold_pose_home"
    assert events[-1] == "close"


def test_invalid_control_command_keeps_previous_pwm_active(
    monkeypatch,
    capsys,
) -> None:
    events: list[object] = []

    class FakePca9685MotionSink:
        def __init__(self, *, config_dir) -> None:
            pass

        def send(self, command) -> None:
            events.append(("send", command.name))

        def close(self) -> None:
            events.append("close")

    replies = iter(("MOVE", "pwm 1000", "release"))
    monkeypatch.setattr(
        main_module,
        "Pca9685MotionSink",
        FakePca9685MotionSink,
    )

    exit_code = main(
        ["control", "--hardware"],
        input_fn=lambda _: next(replies),
    )

    output = capsys.readouterr()
    assert exit_code == 0
    assert events == [("send", "hold_pose_home"), "close"]
    assert "The previous PWM remains active." in output.err


def test_control_cartesian_move_remains_calibration_interlocked(
    monkeypatch,
    capsys,
) -> None:
    events: list[object] = []

    class FakePca9685MotionSink:
        def __init__(self, *, config_dir) -> None:
            pass

        def send(self, command) -> None:
            events.append(("send", command.name))

        def close(self) -> None:
            events.append("close")

    replies = iter(("MOVE", "move 230 180 60", "release"))
    monkeypatch.setattr(
        main_module,
        "Pca9685MotionSink",
        FakePca9685MotionSink,
    )

    exit_code = main(
        ["control", "--hardware"],
        input_fn=lambda _: next(replies),
    )

    output = capsys.readouterr()
    assert exit_code == 0
    assert events == [("send", "hold_pose_home"), "close"]
    assert "Cartesian hardware motion is blocked" in output.err
