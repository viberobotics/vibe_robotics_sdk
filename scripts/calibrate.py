from viberobotics.motor.motor_controller_manager import MotorControllerManager
from viberobotics.configs.config import load_config

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="sundaya1_real_config_short.yaml", help='Path to config file')
    parser.add_argument('--mode', type=int, default=0, help='Mode: 0 for position control')
    args = parser.parse_args()

    cfg = load_config(args.config)
    motor_manager = MotorControllerManager(cfg.real_config.n_motors, 
                                           cfg.real_config.motor_controllers, 
                                           mode=args.mode, 
                                           calibration_file=cfg.real_config.calibration_file)
    if args.mode == 0:
        inp = input('Motor ids to zero (comma separated, ignore to zero all): ')
        if inp.strip() == '':
            motor_ids = None
        else:
            motor_ids = [int(x) for x in inp.split(',')]
        motor_manager.zero_motors(motor_ids)
    else:
        motor_manager.calibrate()