from viberobotics.policy.policy import Policy
from viberobotics.configs.config import SundayA1Config, load_config
from viberobotics.web.controller_web_server import ControllerWebServer
from viberobotics.constants import ControlMode, CONFIG_DIR
from viberobotics.utils.smoothing import EMASmoothing
from viberobotics.utils.pid import PIDController
from viberobotics.utils.math import *

import mujoco, mujoco.viewer
import numpy as np
from abc import ABC, abstractmethod
import time


class SimController:
    def __init__(self, config: SundayA1Config):
        self.config = config
        self.sim_config = config.sim_config
        self.default_qpos = config.default_qpos
        
        self.policy = Policy(config)
        self.controller_server = ControllerWebServer(initial_mode=ControlMode.PD_STAND)
        self.controller_server.start_server()
        
        self._setup_mujoco()
        
        self.kp_torque = self.config.control_config.kp_torque
        self.kd_torque = self.config.control_config.kd_torque
    
        self.mode_map = {
            ControlMode.NONE: self.default_controller,
            ControlMode.PD_STAND: self.pd_stand_controller,
            ControlMode.RL: self.rl_controller
        }
        
        self.smoothed_target = EMASmoothing(0.2, default_value=self.default_qpos)
        self.motor_pd_controller = PIDController(
            kp=self.kp_torque,
            ki=0,
            kd=self.kd_torque,
        )
        
        self.qpos = np.zeros(self.joint_count, dtype=np.float32)
        self.qvel = np.zeros(self.joint_count, dtype=np.float32)
        
        self.next_inference_time = time.perf_counter()
    
    def _setup_mujoco(self):
        self.mj_model = mujoco.MjModel.from_xml_path(self.sim_config.asset_path)
        self.mj_model.opt.timestep = self.sim_config.dt
        self.mj_data = mujoco.MjData(self.mj_model)
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        self.joint_count = self.mj_data.qpos[7:].shape[0]
        self.mj_data.qpos[7:] = self.config.default_qpos.copy()
        mujoco.mj_forward(self.mj_model, self.mj_data)
    
    def default_controller(self):
        return None
    
    def pd_stand_controller(self):
        return self.default_qpos
        
    
    def rl_controller(self):
        if time.perf_counter() < self.next_inference_time:
            return
        self.next_inference_time = time.perf_counter() + self.config.policy_config.policy_interval
        
        quat = self.mj_data.sensor("orientation").data[[1, 2, 3, 0]].astype(np.float32)
        rpy = quat_2_rpy(quat)
        projected_gravity = rotate_vector_inverse_rpy(
            rpy[0], rpy[1], rpy[2],
            np.array([0., 0., -1.])
        )
        base_ang_vel = self.mj_data.sensor("gyro").data.astype(np.float32)
        target_qpos = self.policy.inference(
            dof_pos=self.qpos,
            dof_vel=self.qvel,
            base_ang_vel=base_ang_vel,
            projected_gravity=projected_gravity,
            vx=self.controller_server.get_control_input()[0],
            vy=self.controller_server.get_control_input()[1],
            vyaw=self.controller_server.get_control_input()[2]
        )
        
        return target_qpos
    
    def run(self):
        render_timer = time.perf_counter()
        with mujoco.viewer.launch_passive(self.mj_model, self.mj_data) as viewer:
            viewer.cam.elevation = -20
            while viewer.is_running():
                self.qpos = self.mj_data.qpos[7:].copy().astype(np.float32)
                self.qvel = self.mj_data.qvel[6:].copy().astype(np.float32)
                
                mode = self.controller_server.get_control_mode()
                controller_fn = self.mode_map.get(mode, self.default_controller)
                target_qpos = controller_fn()
                
                if target_qpos is not None:
                    smoothed_target = self.smoothed_target.apply(target_qpos)
                    self.mj_data.ctrl[:] = np.clip(
                        self.motor_pd_controller.update(
                            smoothed_target,
                            self.qpos,
                            derivative=-self.qvel
                        ),
                        self.mj_model.actuator_ctrlrange[:, 0],
                        self.mj_model.actuator_ctrlrange[:, 1]
                    )
                
                mujoco.mj_step(self.mj_model, self.mj_data)
                if time.perf_counter() - render_timer > 1 / 60:
                    render_timer = time.perf_counter()
                    viewer.sync()
                else:
                    time.sleep(self.sim_config.dt)
                    
if __name__ == "__main__":
    config = load_config(CONFIG_DIR / 'sundaya1_sim_config.yaml')
    sim_controller = SimController(config)
    sim_controller.run()