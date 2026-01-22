from viberobotics.configs.config import load_config
from viberobotics.motor.motor_controller_manager import MotorControllerManager

import mujoco, mujoco.viewer
import numpy as np

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default="sundaya1_real_config.yaml", help='Path to config file')
parser.add_argument('--mode', type=int, default=2, help='Mode: 0 for position control, 2 for duty cycle control')
args = parser.parse_args()

cfg = load_config(args.config)
motor_manager = MotorControllerManager(cfg.real_config.motor_controllers, cfg.real_config.motor_order, mode=args.mode)
model = mujoco.MjModel.from_xml_path(cfg.sim_config.asset_path)
data = mujoco.MjData(model)

for i in range(model.njnt):
    # Get the joint name using its ID
    joint_name = model.joint(i).name
    print(f"Joint {i}: {joint_name}")

with mujoco.viewer.launch_passive(model, data) as viewer:
    while True:
        q, dq = motor_manager.get_state()
        mujoco.mj_resetData(model, data)
        data.qpos[:] = np.concatenate([np.array([0, 0, 0., 1, 0, 0, 0]), q])
        # data.qvel[:] = np.concatenate([np.array([0, 0, 0.]), dq])
        mujoco.mj_forward(model, data)
        viewer.sync()