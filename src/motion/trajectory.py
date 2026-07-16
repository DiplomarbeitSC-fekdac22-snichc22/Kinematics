from numpy.ma.core import ceil

def generate_frames(
        start: dict[str, int],
        target: dict[str, int],
        max_step_us: int
) -> list[dict[str, int]]:
    """Generate frames in which all moving joints finish together."""
    if max_step_us <= 0:
        raise ValueError('max_step_us must be greater than 0')

    if start.keys() != target.keys():
        raise ValueError('start and target must have the same joints')

    largest_change = max(
        (
            abs(target[joint] - start[joint])
            for joint in target
        ),
        default=0
    )

    frame_count = max(1, ceil(largest_change / max_step_us))

    frames: list[dict[str, int]] = []

    for frame_number in range(1, frame_count + 1):
        if frame_number == frame_count:
            frames.append(target)
            continue

        frame: dict[str, int] = {}

        for joint, target_value in target.items():
            start_value = start[joint]
            change = target_value - start_value

            value = round(start_value + change * frame_number / frame_count)

            if change > 0:
                value = min(value, target_value - 1)
            elif change < 0:
                value = max(value, target_value + 1)

            frame[joint] = value

        frames.append(frame)

    return frames
