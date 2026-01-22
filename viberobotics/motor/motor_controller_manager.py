from viberobotics.motor.motor_controller import MotorController
from viberobotics.configs.config import MotorControllerConfig
from viberobotics.utils.math import *
from viberobotics.constants import CALIBRATION_FILE


import numpy as np
from typing import List


SIGN_CHANGE_MOTORS = np.array([1,
                               1, 1, 1, 1, -1, 
                               1, -1, -1, -1, 1, 1,
                               -1, 1, 1, 1, 1,
                               1, -1, -1, 1, 1, 1])

class MotorControllerManager:
    def __init__(self, n_motors, motor_mapping: list[MotorControllerConfig], calibration_file=None, mode=0):
        self.motor_mapping = motor_mapping
        self.controllers_mapping: dict[str, MotorController] = {}
        self.controllers: List[MotorController] = []
        self.motor_ids: List[int] = []
        self.n_motors = n_motors
        motor_order = {}
        sign_change = []
        for motor_cfg in motor_mapping:
            controller = MotorController(motor_cfg.motor_ids, motor_cfg.serial_config.port)
            self.controllers_mapping[motor_cfg.name] = controller
            self.controllers.append(controller)
            self.motor_ids.extend(motor_cfg.motor_ids)
            motor_order.update({
                real_id: sim_id
                for real_id, sim_id in zip(motor_cfg.motor_ids, motor_cfg.sim_idxs)
            })
            sign_change.extend(motor_cfg.sign_change)
        sign_change = np.array(sign_change)
        self.motor_ids = np.array(self.motor_ids)
        
        self.motor_order = np.zeros((max(motor_order.keys()) + 1,), dtype=np.int32)
        for real_id, sim_id in motor_order.items():
            self.motor_order[real_id] = sim_id
        
        
        
        self.calibration = None
        if calibration_file is None:
            calibration_file = CALIBRATION_FILE
        with open(calibration_file, "r", encoding="utf-8") as f:
            all_calibration = []
            # each line is zero pos
            for i, line in enumerate(f):
                all_calibration.append(int(line.strip()))
            self.calibration = np.array(all_calibration)
            print(self.calibration)
        self.mode = mode
        self.set_mode(mode)
        self.sign_change = np.ones((self.n_motors,), dtype=np.int32)
        self.sign_change[self.motor_order[self.motor_ids]] = sign_change
        
    
    def _mj_to_real(self, mj_pos):
        real_pos = np.round(mj_pos / STEP_TO_RAD + self.calibration).astype(np.int32)
        real_pos = real_pos % 4096
        return real_pos
    
    def _real_to_mj(self, real_pos):
        where_wrap = np.abs(real_pos - self.calibration) > 2048
        sign = -np.sign(real_pos - self.calibration)
        unwrap_pos = np.where(where_wrap, real_pos + sign * 4096, real_pos)
        mj_pos = (unwrap_pos - self.calibration) * STEP_TO_RAD
        return mj_pos
    
    def get_raw_state(self):
        q = np.zeros((self.n_motors), dtype=np.float32)
        dq = np.zeros((self.n_motors), dtype=np.float32)

        for controller in self.controllers:
            positions, speeds = controller.receive_raw_motor_states()
            idxs = self.motor_order[controller.motor_ids]
            q[idxs] = positions
            dq[idxs] = speeds
        return q, dq
    
    def get_state(self):
        q, dq = self.get_raw_state()
        mj_dq = dq * STEP_TO_RAD
        mj_q = step2rad(q)
        if self.mode == 2:
            mj_q = self._real_to_mj(q)
        return mj_q * self.sign_change, mj_dq * self.sign_change

    def disable_torque(self):
        for controller in self.controllers:
            controller.disable_torque()
    
    def zero_motors(self):
        for controller in self.controllers:
            controller.zero_motors()
    
    def set_mode(self, mode):
        print(f"Setting motor mode to {mode}")
        self.mode = mode
        for controller in self.controllers:
            controller.set_mode(mode)
            # controller.disable_torque()
            
    def set_duty(self, torques):
        torques *= -self.sign_change
        for controller in self.controllers:
            controller.set_duty(torques[self.motor_order[controller.motor_ids]])
    
    def set_kp_kd(self, kp, kd, overwrite=None):
        if not isinstance(kp, np.ndarray):
            kp = kp * np.ones((self.n_motors,))
        if not isinstance(kd, np.ndarray):
            kd = kd * np.ones((self.n_motors,))
        if overwrite is not None:
            for motor_id, (kp_val, kd_val) in overwrite.items():
                sim_id = self.motor_order[motor_id]
                kp[sim_id] = kp_val
                kd[sim_id] = kd_val
        for controller in self.controllers:
            controller.set_kp_kd(kp, kd)
        
    def set_raw_positions(self, q_pos_step, q_vel, q_acc):
        assert self.mode == 0, "Can only set raw positions in Position Mode"
        for controller in self.controllers:
            idxs = self.motor_order[controller.motor_ids]
            controller.send_raw_positions(q_pos_step[idxs], q_vel[idxs], q_acc[idxs])
        
    def set_positions(self, q_pos, q_vel, q_acc):
        assert self.mode == 0, "Can only set positions in Position Mode"
        q_pos_step = rad2step(q_pos * self.sign_change)
        print(q_pos_step)
        
        if not isinstance(q_vel, np.ndarray):
            q_vel = q_vel * np.ones_like(q_pos_step)
        if not isinstance(q_acc, np.ndarray):
            q_acc = q_acc * np.ones_like(q_pos_step)
            
        self.set_raw_positions(q_pos_step, q_vel, q_acc)
    
    def diff(self, q_target):
        q, dq = self.get_state()
        return q_target - q, dq
    
    def calibrate(self):
        all_q = np.zeros((21,), dtype=np.int32)
        input("Put the robot in the zero position and press Enter to continue...")
        q, _ = self.get_raw_state()
        
        np.savetxt(CALIBRATION_FILE, q, delimiter=",", fmt="%d")