from robot import Robot
from viberobotics.configs.config import load_config
from viberobotics.motor.motor_controller_manager import MotorControllerManager
from viberobotics.utils.utils import get_asset_path
import pygame
from enum import Enum
import numpy as np
import time
from loop_rate_limiters import RateLimiter


class JoystickButton(Enum):
    A = 0
    B = 1
    X = 2
    Y = 3
    LB = 4
    RB = 5
    START = 7

class Demo(Robot):
    def wait_for_button(self, button_id):
        while self.joystick.get_button(button_id) == 0:
            for _ in pygame.event.get():
                pass
    
    def get_button(self, button: JoystickButton):
        for _ in pygame.event.get():
            pass
        return self.joystick.get_button(button.value) == 1
    
    def get_current_button(self):
        for _ in pygame.event.get():
            pass
        for button in JoystickButton:
            if self.joystick.get_button(button.value) == 1:
                return button
        return None
    
    def deploy_controller(self, motor_manager: MotorControllerManager):
        try:
            dt = 0.03
            rate_limiter = RateLimiter(frequency=1 / dt, warn=True)
            self.fsm.start_walking = True
            cmd = np.zeros(3)
            while True:
                current_button = self.get_current_button()
                if current_button is not None:
                    return current_button
                self.fsm.set_cmd(self.get_joystick_cmd())
                print(self.fsm.cmd)
                self.fsm.on_tick()
                
                start_time = time.time()
                self.q, success = self.ik(self._get_targets())
                
                q_full = np.zeros(motor_manager.n_motors)
                q_full[motor_manager.get_sim_idxs('leg')] = self.q[7:]
                motor_manager.set_positions(q_full, 0, 50)
        except KeyboardInterrupt:
            motor_manager.disable_torque()
    
    def teleop(self, 
               leader_motor_manager: MotorControllerManager, 
               follower_motor_manager: MotorControllerManager):
        leader_motor_manager.disable_torque()
        follower_motor_manager.set_positions(np.zeros(follower_motor_manager.n_motors), 0, 30)
        time.sleep(1.)
        raw_zero_q = follower_motor_manager.get_raw_state()[0]
        while True:
            if self.get_button(JoystickButton.START):
                follower_motor_manager.set_positions(np.zeros(follower_motor_manager.n_motors), 0, 30)
                time.sleep(1.)
                break
            q, _ = leader_motor_manager.get_raw_state()
            full_q = raw_zero_q.copy()
            full_q[follower_motor_manager.get_sim_idxs('arm')] = q
            follower_motor_manager.set_raw_positions(full_q, 0, 30)
            time.sleep(0.03)
    
    def run(self):
        cfg = load_config('sundaya1_real_config_short.yaml')
        motor_manager = MotorControllerManager(
            cfg.real_config.n_motors, 
            cfg.real_config.motor_controllers, 
            cfg.real_config.calibration_file, 
            mode=0
        )
        # leader_cfg = load_config('sundaya1_real_config_short_arm.yaml')
        # leader_motor_manager = MotorControllerManager(
        #     leader_cfg.real_config.n_motors, 
        #     leader_cfg.real_config.motor_controllers, 
        #     leader_cfg.real_config.calibration_file, 
        #     mode=0
        # )
        motor_manager.set_positions(cfg.default_qpos, 0, 30)
        self.wait_for_button(JoystickButton.START.value)
        print('starting')
        while True:
            button = self.deploy_controller(motor_manager)
            print(f'Button {button} pressed')
            motor_manager.set_positions(cfg.default_qpos, 0, 30)
            if button == JoystickButton.A:
                motor_manager.play_recording(get_asset_path('motions/waving_motion.json'))
                print('motion done, pressing START to continue')
                self.wait_for_button(JoystickButton.START.value)
            elif button == JoystickButton.B:
                motor_manager.play_recording(get_asset_path('motions/lay_down_motion.json'))
                will_exit = True
                while True:
                    if self.get_button(JoystickButton.START):
                        will_exit = False
                        break
                    elif self.get_button(JoystickButton.X):
                        break
                    time.sleep(0.1)
                if will_exit:
                    print('end demo')
                    break
                else:
                    motor_manager.play_recording(get_asset_path('motions/stand_up_motion.json'))
            # elif button == JoystickButton.Y:
            #     self.teleop(leader_motor_manager, motor_manager)
            #     pass
    
if __name__ == '__main__':
    demo = Demo()
    demo.run()
        