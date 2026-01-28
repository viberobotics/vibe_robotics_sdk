from viberobotics.motor.motor_controller_manager import MotorControllerManager
from viberobotics.configs.config import load_config

import argparse
import numpy as np
from numpy import array

if __name__ == '__main__':
    config_name = 'sundaya1_real_config_arm_only.yaml'
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
        array([ 0.36662149,  0.82067966, -1.25633025, -0.72250485, -0.04295135,
        0.00766993, -0.00306797,  0.00460196,  0.00306797, -0.00306797,
       -0.00153399]),
        array([ 0.35741758,  0.82067966, -1.25633025, -0.72403884,  0.96027207,
        0.00766993, -0.00306797,  0.00460196,  0.00306797, -0.00306797,
       -0.00153399]),
        array([ 0.35895157,  0.82221365, -1.25633025, -0.72557282,  0.26231074,
        0.00766993, -0.00306797,  0.00460196,  0.00306797, -0.00306797,
       -0.00153399]),
        array([-0.02914572, -0.92499042, -1.25633025,  0.93879628,  0.26077676,
        0.00766993, -0.00306797,  0.00460196,  0.00306797, -0.00306797,
       -0.00153399]),
        array([-0.02914572, -0.92345643, -1.25633025,  0.91885448,  0.96487403,
        0.00766993, -0.00306797,  0.00460196,  0.00306797, -0.00306797,
       -0.00153399])
    ]
    
    # motor_manager.disable_torque()
    # while True:
    #     input('>')
    #     q = motor_manager.get_state()[0]
    #     print('Current positions:', repr(q))
    
    for pose in poses:
        input('>')
        motor_manager.set_positions(pose, 0, 30)
    
    motor_manager.disable_torque()