from pathlib import Path
from threading import Event

from config.config_loader import DEFAULT_CONFIG_DIR, load_config
from motion.trajectory import generate_frames
from state_machine.pick_and_place import MotionCommandSink


class MotionCancelled(RuntimeError):
    """Raised when the emergency-stop flag cancels movement."""


class MotionExecutionError(RuntimeError):
    """Wrap a sink failure with trajectory-frame context."""
    def __init__(
            self,
            *,
            command_name: str,
            frame_number: int,
            frame_count: int,
            pulses_us: dict[str, int],
            cause: Exception,
    ) -> None:
        self.command_name = command_name
        self.frame_number = frame_number
        self.frame_count = frame_count
        self.pulses_us = dict(pulses_us)
        self.cause = cause
        super().__init__(
            f"Motion {command_name!r} failed at frame "
            f"{frame_number}/{frame_count}: "
            f"{type(cause).__name__}: {cause}"
        )


class MotionExecutor:
    """Blocking trajectory executor implementing MotionCommandSink."""
    def __init__(
            self,
            sink: MotionCommandSink,
            *,
            emergency_stop: Event | None = None,
            initial_pulses_us: dict[str, int] | None = None,
            config_dir: Path | str = DEFAULT_CONFIG_DIR,
    ) -> None:
        config = load_config(
            "kinematics_settings.toml",
            config_dir,
        )["motion_interpolation"]

        self.sink = sink
        self.emergency_stop = emergency_stop or Event()

        self.frame_interval_s = float(config["step_time_s"])
        self.max_step_us = int(config["max_step_pwm_us"])

        if not 0.02 <= self.frame_interval_s <= 0.03:
            raise ValueError("Frame interval must be between 0.02 and 0.03 seconds")

        self.last_pulses_us = dict(initial_pulses_us or {})

    def send(self, command) -> None:
        """Send all frames and return only after the target frame."""
        target = dict(command.pulses_us)

        # A command may change only one joint, such as the gripper.
        start = {
            joint: self.last_pulses_us.get(joint, target_value)
            for joint, target_value in target.items()
        }

        frames = generate_frames(
            start,
            target,
            self.max_step_us,
        )

        for frame_number, pulses in enumerate(frames, start=1):
            if self.emergency_stop.is_set():
                raise MotionCancelled(
                    f"Motion {command.name} was cancelled"
                )

            final_frame = frame_number == len(frames)

            if final_frame:
                frame_command = command
            else:
                frame_command = type(command)(
                    name=command.name,
                    pulses_us=pulses,
                )

            try:
                self.sink.send(frame_command)
            except Exception as exc:
                raise MotionExecutionError(
                    command_name=command.name,
                    frame_number=frame_number,
                    frame_count=len(frames),
                    pulses_us=pulses,
                    cause=exc,
                ) from exc

            self.last_pulses_us.update(pulses)

            if not final_frame:
                cancelled = self.emergency_stop.wait(
                    self.frame_interval_s
                )

                if cancelled:
                    raise MotionCancelled(
                        f"Motion {command.name} was cancelled"
                    )

    def cancel(self) -> None:
        self.emergency_stop.set()

    def clear_cancel(self) -> None:
        self.emergency_stop.clear()