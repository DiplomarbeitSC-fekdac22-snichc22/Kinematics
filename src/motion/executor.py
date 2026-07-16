class MotionCancelled(RuntimeError):
    """Raised when the emergency-stop flag cancels movement."""

class MotionExecutor:
    """Blocking trajectory executor implementing MotionCommandSink."""

    