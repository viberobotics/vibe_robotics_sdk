from viberobotics.motor.motor_controller_manager import MotorControllerManager
from viberobotics.configs.config import load_config
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default="sundaya1_real_config_leg_only.yaml", help='Path to config file')
args = parser.parse_args()

config = load_config(args.config)
motor_manager = MotorControllerManager(config.real_config.motor_controllers, mode=0)
motor_manager.disable_torque()