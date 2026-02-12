from viberobotics.motor.motor_controller_manager import MotorControllerManager
from viberobotics.configs.config import load_config

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="sundaya1_real_config.yaml", help='Path to config file')
    parser.add_argument('--mode', type=int, default=0, help='Mode: 0 for position control')
    args = parser.parse_args()

    cfg = load_config(args.config)
    motor_manager = MotorControllerManager(cfg.real_config.n_motors, 
                                           cfg.real_config.motor_controllers, 
                                           mode=args.mode, 
                                           calibration_file=cfg.real_config.calibration_file)
    if args.mode == 0:
        motor_manager.zero_motors()
    else:
        motor_manager.calibrate()