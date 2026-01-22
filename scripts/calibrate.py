from viberobotics.motor.motor_controller_manager import MotorControllerManager
from viberobotics.constants import CONFIG_DIR
from viberobotics.configs.config import MotorControllerConfig, SerialConfig

import numpy as np
import sys
import os
import json
from dataclasses import dataclass

CACHE_PATH = "/tmp/viberobotics_calibrate_cache.json"

def load_cache():
    if not os.path.exists(CACHE_PATH):
        return {}
    try:
        with open(CACHE_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def save_cache(cache):
    try:
        with open(CACHE_PATH, "w") as f:
            json.dump(cache, f)
    except Exception:
        pass


def prompt_with_default(prompt, default):
    if default is not None and default != "":
        prompt = f"{prompt} [{default}]: "
    else:
        prompt = f"{prompt}: "
    value = input(prompt).strip()
    if value == "" and default is not None:
        return str(default)
    return value

mode = prompt_with_default(
    "Select calibration mode (0: Position, 2: PWM)",
    "2",
)

cache = load_cache()
n_motor_controllers = prompt_with_default(
    "Enter number of motor controllers",
    cache.get("n_motor_controllers"),
)
try:
    n_motor_controllers = int(n_motor_controllers)
except:
    print("Invalid number of motor controllers.")
    sys.exit(1)

motor_mapping = []
motor_idxs = []
motor_order = {}
cached_controllers = cache.get("controllers", [])
for i in range(n_motor_controllers):
    cached = cached_controllers[i] if i < len(cached_controllers) else {}
    port = prompt_with_default(
        f"Enter port for motor controller {i+1} (e.g., /dev/ttyUSB0)",
        cached.get("port"),
    )
    motor_ids_str = prompt_with_default(
        f"Enter motor IDs for motor controller {i+1} (comma-separated)",
        cached.get("motor_ids"),
    )
    motor_sim_idx_str = prompt_with_default(
        f"Enter motor sim indices for motor controller {i+1} (comma-separated)",
        cached.get("sim_indices"),
    )
    try:
        motor_ids = [int(mid.strip()) for mid in motor_ids_str.split(",")]
    except:
        print("Invalid motor IDs.")
        sys.exit(1)
    try:
        sim_indices = [int(sid.strip()) for sid in motor_sim_idx_str.split(",")]
    except:
        print("Invalid motor sim indices.")
        sys.exit(1)
    for mid, sid in zip(motor_ids, sim_indices):
        motor_order[mid] = sid
    motor_mapping.append(MotorControllerConfig(
        name=f"motor_controller_{i+1}",
        motor_ids=motor_ids,
        serial_config=SerialConfig(
            port=port,
            baudrate=1000000,
        ),
    ))

save_cache(
    {
        "n_motor_controllers": n_motor_controllers,
        "controllers": [
            {"port": m.serial_config.port, "motor_ids": ",".join(map(str, m.motor_ids)), "sim_indices": ",".join(str(motor_order[mid]) for mid in m.motor_ids)}
            for m in motor_mapping
        ],
    }
)
mode = int(mode)
manager = MotorControllerManager(motor_mapping, motor_order=motor_order,calibration_file=None, mode=mode)
if mode == 0:
    manager.zero_motors()
    print('motor zeroed')
elif mode == 2:
    manager.calibrate()
