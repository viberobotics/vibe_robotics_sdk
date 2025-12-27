# Viberobotics Python Code Catalog

Repository code organized by folder with brief notes on each file.

## Root
- `setup.py` — setuptools configuration for the `viberobotics` package.
- `LICENSE` — Apache 2.0 license.
- `viberobotics.egg-info/` — generated package metadata (PKG-INFO, SOURCES, etc.).

## viberobotics (package root)
- `constants.py` — shared enums and paths (`ControlMode`, `ASSET_DIR`, `CONFIG_DIR`).
- `__pycache__/` — Python bytecode caches.

### configs
- `config.py` — dataclasses for simulation, control, and policy configuration plus YAML loader that resolves asset/model paths.
- `sundaya1_sim_config.yaml` — sample config with default positions, control gains, and MuJoCo asset/model paths.
- `__pycache__/` — bytecode caches.

### controller
- `deploy_sim.py` — launches MuJoCo passive viewer, wires keyboard/web control into controllers (default/PD stand/RL), drives policy actions, and steps simulation loop.
- `__pycache__/` — bytecode cache.

### policy
- `policy.py` — loads TorchScript policy, maintains observation buffer and rate-limited command smoothing, converts between policy and MuJoCo joint indexing, and returns target joint positions.
- `__pycache__/` — bytecode cache.

### utils
- `pid.py` — simple PID controller with derivative option.
- `smoothing.py` — abstract smoothing filter plus rate-limited and EMA implementations.
- `buffer.py` — fixed-size observation buffer with rolling window flattening.
- `math.py` — utility math for angle-step conversions and roll/pitch/yaw rotations.
- `__pycache__/` — bytecode caches.

### web
- `controller_web_server.py` — threaded HTTP server exposing control state endpoints (/status, /state, /mode, /release_all); shares state with handler for keyboard/web UI.
- `controller_client.html` — browser UI for mode selection and WASD/QE vector control; fetches `/status` for initial state and posts updates to the server.
- `__pycache__/` — bytecode cache.

### motor
- `motor_controller.py` — high-level FT servo motor controller (port init, grouped reads of position/speed/current/load, duty writes, torque-off zeroing); uses SDK helpers.
- `ftservo_python_sdk/` — vendor SDK for Feetech/FT servos (setup files, LICENSE, README); contains protocol handlers, port access, and SCSCL/SMS_STS command helpers plus egg-info metadata.

### exceptions
- `motor.py` — custom exceptions for motor communication failures (group add/read/write errors, availability, sync write issues).

### assets
- `assets/models/policy_leo.pt` — TorchScript policy checkpoint consumed by `policy.py`.
- `assets/mujoco/` — MuJoCo scene/model XMLs and geometry assets for SundayA1 variants (STL/PART files, config JSON).

