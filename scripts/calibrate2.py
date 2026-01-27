from viberobotics.motor.motor_controller_manager import MotorControllerManager
from viberobotics.configs.config import load_config

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="sundaya1_real_config_half_2dof.yaml", help='Path to config file')
    args = parser.parse_args()

    cfg = load_config(args.config)
    motor_manager = MotorControllerManager(cfg.real_config.n_motors, 
                                           cfg.real_config.motor_controllers, 
                                           mode=0, 
                                           calibration_file=cfg.real_config.calibration_file)
    motor_manager.calibrate()