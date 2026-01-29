from viberobotics.motor.motor_controller_manager import MotorControllerManager
from viberobotics.configs.config import load_config

import argparse
import numpy as np
from numpy import array

if __name__ == '__main__':
    config_name = 'sundaya1_real_config.yaml'
    config = load_config(config_name)
    
    motor_manager = MotorControllerManager(
        n_motors=config.real_config.n_motors,
        motor_mapping=config.real_config.motor_controllers,
        calibration_file=config.real_config.calibration_file,
        mode=0
    )
    motor_manager.set_positions(config.default_qpos, 0, 30)
    
    input('start>')
    
    
    poses = [
        
    ]
    
    motor_manager.controllers_mapping['arm'].disable_torque()
    while True:
        x = input('>')
        if x == 'q':
            break
        q = motor_manager.get_state()[0]
        print('Current positions:', repr(q))
        poses.append(q)
    
    for pose in poses:
        input('>')
        motor_manager.set_positions(pose, 0, 30)