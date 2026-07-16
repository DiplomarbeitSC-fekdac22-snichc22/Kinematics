from pathlib import Path
from threading import Event

from config.config_loader import DEFAULT_CONFIG_DIR, load_config
from motion.trajectory import generate_frames
from state_machine.pick_and_place import MotionCommandSink


class MotionCancelled(RuntimeError):
    """Raised when the emergency-stop flag cancels movement."""


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

        for index, pulses in enumerate(frames):
            if self.emergency_stop.is_set():
                raise MotionCancelled(
                    f"Motion {command.name} was cancelled"
                )

            final_frame = index == len(frames) - 1

            if final_frame:
                frame_command = command
            else:
                frame_command = type(command)(
                    name=command.name,
                    pulses_us=pulses,
                )

            self.sink.send(frame_command)

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