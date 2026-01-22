from viberobotics.motor.motor_controller_manager import MotorControllerManager
from viberobotics.configs.config import load_config
from viberobotics.constants import CALIBRATION_FILE
from viberobotics.utils.pid import PIDController

from pprint import pprint
import numpy as np
import sys

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=int, default=2, help='Mode: 0 for position control, 2 for duty cycle control')
parser.add_argument('--config', type=str, default="sundaya1_real_config_leg_only.yaml", help='Path to config file')
args = parser.parse_args()

mode = args.mode
config = load_config(args.config)
pprint(config)
if mode == 2:
    kp = 500
    kd = 50

    motor_manager = MotorControllerManager(config.real_config.motor_controllers, config.real_config.motor_order, calibration_file=CALIBRATION_FILE, mode=2)
    default_qpos = config.default_qpos
    controller = PIDController(kp, 0, kd)
    print(default_qpos)
    while True:
        q, dq = motor_manager.get_state()
        with np.printoptions(suppress=True, precision=6):
            print(q)
        duty = controller.update(
            setpoint=default_qpos,
            measurement=q,
            derivative=-dq,
        )
        motor_manager.set_duty(duty)
        print(duty)
elif mode == 0:
    kp = 32
    kd = 32
    cfg = config.real_config.motor_controllers
    motor_manager = MotorControllerManager(cfg, calibration_file=CALIBRATION_FILE, mode=0)
    motor_manager.set_kp_kd(kp, kd)
    default_qpos = config.default_qpos
    motor_manager.set_positions(default_qpos, 0, 20)