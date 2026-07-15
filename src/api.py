from state_machine.pick_and_place import (
    MotionCommandSink,
    PickAndPlaceStateMachine,
    TargetPosition,
)


class RobotController:
    def __init__(self, motion_sink: MotionCommandSink) -> None:
        self.machine = PickAndPlaceStateMachine(sink=motion_sink)

    def start_pick_and_place(
            self,
            x_mm: float,
            y_mm: float,
            z_mm: float,
    ) -> None:
        self.machine.start_pick_and_place(
            TargetPosition(
                x_mm=x_mm,
                y_mm=y_mm,
                z_mm=z_mm,
            )
        )

    def run_pick_and_place(
            self,
            x_mm: float,
            y_mm: float,
            z_mm: float,
    ) -> bool:
        """Start one complete blocking pick-and-place sequence."""
        self.start_pick_and_place(x_mm, y_mm, z_mm)
        return self.machine.run_until_finished()
