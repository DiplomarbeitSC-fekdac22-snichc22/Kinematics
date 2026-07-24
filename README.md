# Robot Arm Kinematics

## Kinematics progress

**Estimated overall completion:**

`[############--------] 60%`

_Last reviewed: 2026-07-16. This is a readiness estimate, not a test-coverage metric. The mathematical model is substantially implemented; physical calibration and end-to-end robot validation remain the largest gaps._

| Area                                           | Progress | Current state                                                                        |
|------------------------------------------------|---------:|--------------------------------------------------------------------------------------|
| Robot geometry and coordinate frames           |      60% | CAD-derived model and                                                                |
| Inverse and forward kinematics                 |      85% | J1-J4 solver, gripper offset and FK round-trip checks implemented                    |
| Workspace and shelf validation                 |      80% | Free workcell and two shelf compartments represented                                 |
| Angle-to-PWM and PCA9685 output                |      70% | Conversion, channel mapping and safe-range enforcement implemented                   |
| Pick-and-place sequencing                      |      75% | Validation, approach, grasp, lift, retract, deposit and home states implemented      |
| Webots simulation                              |      70% | Workcell, robot, sensors, adapter and smoke-test scripts implemented                 |
| Motion interpolation and cancellation          |      50% | Frame generator and emergency-stop-aware executor implemented; more tests are needed |
| Physical calibration and real-robot validation |      15% | Servo zeros, directions, limits, poses and gripper pulses are still provisional      |

**Current physical-hardware blocker:** the PCA9685 channel assignment and several named-pose pulses are recorded, but the mathematical angle-to-pulse calibration for J1-J4 is still provisional. Arbitrary Cartesian hardware motion is therefore interlocked; dry runs and guarded playback of recorded poses remain available.

## Purpose

This repository contains the kinematics and motion-control subsystem for a ceiling-mounted, servo-driven pick-and-place robot arm. Upstream vision software is expected to provide an object position in robot coordinates. This project then:

1. validates the target and workspace constraints;
2. calculates joint angles with inverse kinematics;
3. converts joint angles to PWM pulse widths;
4. executes a pick-and-place state machine;
5. sends the same motion commands to a dry-run logger, Webots or a PCA9685 servo driver.

```text
Target XYZ
   -> workspace/reachability checks
   -> inverse kinematics
   -> joint and pulse validation
   -> pick-and-place state machine
   -> DryRun / JSON / Webots / PCA9685
```

## Coordinate system

All public target positions use millimetres in the robot frame:

| Axis | Positive direction |
|---|---|
| `X` | forward / depth |
| `Y` | downward from the top of the workcell |
| `Z` | right / lateral |

Angles are in degrees and servo pulse widths are in microseconds.

Example target: `x=230 mm, y=180 mm, z=60 mm`.

## Installation

Requirements: Python 3.11 or newer on Linux. Webots R2025a and Raspberry Pi hardware support are optional.

```bash
git clone https://github.com/DiplomarbeitSC-fekdac22-snichc22/Kinematics.git
cd Kinematics

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

For Raspberry Pi and PCA9685 support:

```bash
python -m pip install -e ".[dev,hardware]"
```

## Usage

### Validate configuration and run tests

After resolving the current configuration blockers:

```bash
python -m config.config_checker
python simulation/scripts/validate_project.py
python -m pytest
```

### Run fault-injection tests

The deterministic fault suite covers state-machine command failures,
mid-trajectory transport failures, emergency-stop cancellation, ambiguous
after-send delivery, PCA9685 partial writes, and best-effort output shutdown.

```bash
python -m pytest \
  tests/test_fault_injection.py \
  tests/test_pick_and_place_faults.py \
  tests/test_motion_executor_faults.py \
  tests/test_pca9685_faults.py
```

Fault rules can target a command name, a one-based sink call number, or both.
`before_send` means the wrapped sink did not receive the command. `after_send`
models an ambiguous acknowledgement failure after the wrapped sink accepted it.
The full design and safety invariants are documented in
`docs/fault_injection_testing.md`.

Print the main geometry, pose and channel settings:

```bash
python examples/print_config_summary.py
```

### Calculate inverse kinematics

```bash
python -m kinematics.inverse_kinematics 230 180 60
```

The command prints J1-J4 angles, estimated PWM values, reachability and rejection reasons.

Forward kinematics is available through:

```python
from kinematics.forward_kinematics import calculate_gripper_center

position = calculate_gripper_center({
    "J1_base": 0.0,
    "J2_shoulder": -90.0,
    "J3_elbow": 150.0,
    "J4_wrist": 60.0,
})
```

### Run a dry pick-and-place sequence

```bash
python -m state_machine.pick_and_place
```

This uses the console-only sink and writes the generated sequence to `pick_and_place_commands.json`.

The public API can be embedded in another application:

```python
from api import RobotController
from state_machine.pick_and_place import DryRunMotionSink

controller = RobotController(DryRunMotionSink())
success = controller.run_pick_and_place(230.0, 180.0, 60.0)
```

`RobotController` generates and validates the complete nine-waypoint motion
plan before it creates the execution state machine. If any future waypoint is
invalid, `run_pick_and_place()` returns `False`,
`controller.last_planning_failure` contains the structured reason, and no
motion command is sent. Dry-run and simulation sinks report commissioning-range
violations as planning warnings; the PCA9685 path rejects them during planning.

Available motion sinks:

| Sink | Use |
|---|---|
| `DryRunMotionSink` | Print commands without moving hardware |
| `JsonRecordingMotionSink` | Record commands and optionally wrap another sink |
| `WebotsMotionSink` | Execute the same commands in Webots |
| `Pca9685MotionSink` | Send verified pulse widths to the real servo driver |
| `MotionExecutor` | Interpolate PWM frames and support cancellation around another sink |

### Manual operation without a backend

Installing the project creates the `robot-arm` command. Every movement is a
dry run unless `--hardware` is supplied.

```bash
# Show channel mapping, recorded poses, and calibration blockers
robot-arm check

# Inspect a recorded pose without moving hardware
robot-arm pose ready

# Plan one gripper-centre move
robot-arm move 230 180 60 --from-pose home

# Omit X Y Z to enter them interactively
robot-arm move --from-pose home
```

Recorded poses can be commissioned before the mathematical calibration is
complete because they use the measured pulse values in `poses.toml`:

```bash
robot-arm pose ready --hardware --from-pose home
```

The physical arm must actually match `--from-pose` before servo outputs are
enabled. The command requires a typed confirmation, returns to the declared
start pose, and waits for separate servo power to be switched off before it
disables PCA9685 outputs.

Real XYZ or pick-and-place commands remain blocked until J1-J4 have measured
angle/pulse fits and assembled limits, every corresponding
`requires_physical_calibration` value is `false`, and
`hardware_cartesian_motion_enabled = true` has been set in
`configs/servo_calibration.toml`.

```bash
robot-arm move 230 180 60 --hardware --from-pose home
robot-arm pick 230 180 60 --hardware --from-pose home
```

For persistent open-loop control without a backend, start one control session:

```bash
robot-arm control --hardware
```

After the one-time `MOVE` confirmation, the session immediately commands the
recorded `home` PWM. The PCA9685 repeats the selected PWM at 50 Hz until another
command is entered:

```text
pose ready
pose deposit
pwm 977 1743 2134 1172 732
status
home
release
```

The five direct PWM values are ordered as `J1_base J2_shoulder J3_elbow
J4_wrist J5_gripper`. Every value is checked against the configured hardware
range, then written directly without trajectory interpolation. An invalid
command leaves the previous PWM active. `release`, end-of-input, or `Ctrl+C`
disables all PCA9685 outputs and ends the session.

Once Cartesian hardware motion is calibrated and enabled, the same session
also accepts:

```text
move 230 180 60
```

This mode tracks the last commanded state, not the measured physical state.
If a servo stalls, slips, loses power, or is moved externally, the assumption
is no longer valid. Enter `home` to command the known PWM again.

[//]: # (### Run Webots)

[//]: # ()

[//]: # (Set `WEBOTS_HOME` or place the `webots` executable on `PATH`.)

[//]: # ()

[//]: # (```bash)

[//]: # (# Static project/configuration checks)

[//]: # (python simulation/scripts/validate_project.py)

[//]: # ()

[//]: # (# Open the simulation)

[//]: # (./simulation/scripts/run_webots.sh)

[//]: # ()

[//]: # (# Headless fast smoke test)

[//]: # (./simulation/scripts/smoke_test_webots.sh)

[//]: # (```)

[//]: # ()

[//]: # (The default simulation target is stored in `controllerArgs` inside `simulation/worlds/robot_arm_pick_and_place.wbt`.)

### Explore the workspace

```bash
python tools/workspace_roulette.py
```

`tools/visualize_workspace.ipynb` provides a notebook-based workspace visualization. Notebook dependencies are not part of the core installation.

### Use real PCA9685 hardware

The intended stack is:

```text
RobotController
   -> MotionExecutor
   -> Pca9685MotionSink
   -> PCA9685
   -> servos
```

Do not use the hardware sink until every value marked as provisional has been measured. Keep a hardware emergency stop that cuts servo power independently of Python and the Raspberry Pi.

The PCA9685 logic side uses the Raspberry Pi's default I2C pins and address
`0x40`. Power the servos from their separate fused supply, not from the
Raspberry Pi. Only one controller may drive the PCA9685 at a time.

## Configuration

| File | Purpose |
|---|---|
| `configs/robot_geometry.toml` | CAD-derived joint centres, link lengths and gripper geometry |
| `configs/kinematics_settings.toml` | Coordinate convention, IK options, workspace, shelves and motion settings |
| `configs/servo_calibration.toml` | Joint roles, angle limits, direction, zero and pulse conversion |
| `configs/pca9685.toml` | PWM frequency, conversion constants and output channels |
| `configs/poses.toml` | Named poses, deposit target and gripper commands |
| `configs/physical_measurements_required.toml` | Remaining real-world measurement checklist |
| `configs/webots_simulation.toml` | Webots devices, transforms, timing and actuator parameters |

## Repository layout

```text
configs/       Robot, servo, pose and simulation configuration
src/kinematics IK, FK, PWM conversion and workspace checks
src/planning/ Complete waypoint generation and prevalidation
src/state_machine
               Execution of accepted plans and command sinks
src/motion/    Interpolation and cancellable execution
src/hardware/  PCA9685 backend
src/simulator/ Webots coordinate and motion adapters
simulation/    Webots world, PROTOs, controller and scripts
tests/         Regression and adapter tests
tools/         Workspace exploration and visualization
```

## Still missing

- **Configuration:** duplicate TOML table · `grasp_depth_offset_mm`
- **Calibration:** joint zeros · direction signs · mechanical limits · safe pulse limits · home/ready/deposit poses · gripper open/close/hold pulses
- **Measurements:** closed gripper width · object dimensions · grip point · collision zones · cable limits
- **Validation:** real FK/IK error measurements · repeatability · payload tests · shelf-row trials · full end-to-end run
- **Motion:** trajectory/executor tests · acceleration profile · collision-aware path planning
- **Safety:** hardware E-stop integration · limit/home switches · fault recovery
- **Sensors:** camera target interface · ToF approach feedback · grasp confirmation · closed-loop correction
- **Engineering:** CI workflow · backend/service integration · deployment service · release/versioning process

## Safety

This is an alpha-stage educational robotics project. The current servo calibration is explicitly provisional and is not safe for unattended operation. Use separate servo power, conservative limits, a hardware emergency stop and a clear work area during every physical test.

## License

Licensed under the GNU General Public License v3.0. See `LICENSE`.
