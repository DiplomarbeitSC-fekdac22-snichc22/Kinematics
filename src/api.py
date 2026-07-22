from planning.models import MotionPlan, PlanningFailure, TargetPose
from planning.pick_and_place_planner import PickAndPlacePlanner
from state_machine.pick_and_place import MotionCommandSink, PickAndPlaceStateMachine


def _requires_hardware_safe_prevalidation(sink: object) -> bool:
    """Follow common sink wrappers to find a real hardware boundary."""
    seen: set[int] = set()
    current: object | None = sink
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if bool(
            getattr(
                current,
                "requires_hardware_safe_prevalidation",
                False,
            )
        ):
            return True
        current = getattr(
            current,
            "sink",
            getattr(current, "wrapped_sink", None),
        )
    return False


class RobotController:
    def __init__(
        self,
        motion_sink: MotionCommandSink,
        *,
        planner: PickAndPlacePlanner | None = None,
    ) -> None:
        self.motion_sink = motion_sink
        self.planner = planner or PickAndPlacePlanner(
            enforce_hardware_safe_limits=(
                _requires_hardware_safe_prevalidation(motion_sink)
            )
        )
        self.machine: PickAndPlaceStateMachine | None = None
        self.last_plan: MotionPlan | None = None
        self.last_planning_failure: PlanningFailure | None = None

    def start_pick_and_place(
            self,
            x_mm: float,
            y_mm: float,
            z_mm: float,
    ) -> bool:
        """Plan the whole sequence before constructing its state machine."""
        result = self.planner.plan(
            TargetPose(x_mm=x_mm, y_mm=y_mm, z_mm=z_mm)
        )
        if isinstance(result, PlanningFailure):
            self.last_plan = None
            self.last_planning_failure = result
            self.machine = None
            print(
                "[ROBOT] Pick-and-place planning rejected: "
                f"{result.code}: {result.message}"
            )
            return False

        self.last_plan = result
        self.last_planning_failure = None
        self.machine = PickAndPlaceStateMachine(
            sink=self.motion_sink,
            plan=result,
        )
        self.machine.start_pick_and_place()
        return True

    def run_pick_and_place(
            self,
            x_mm: float,
            y_mm: float,
            z_mm: float,
    ) -> bool:
        """Start one complete blocking pick-and-place sequence."""
        if not self.start_pick_and_place(x_mm, y_mm, z_mm):
            return False

        assert self.machine is not None
        return self.machine.run_until_finished()
