from classes import *
from interp import *
from typing import List
from mpc import LinearPredictiveControl
import numpy as np

class WalkingFSM:
    def __init__(self, 
                 ssp_duration, 
                 dsp_duration, 
                 footsteps: List[Footstep],
                 robot_config: RobotConfig,
                 robot_params: RobotParams):
        self.ssp_duration = ssp_duration
        self.dsp_duration = dsp_duration
        self.footsteps = footsteps
        
        self.state: WalkState = WalkState.STAND
        
        self.start_walking = False
        self.next_footstep = 2
        
        self.dt = 0.03
        self.stance = Stance(
            left_foot=Foot(
                np.array([-robot_params.foot_spred, robot_params.foot_y, 0.]),
                robot_params.foot_size),
            right_foot=Foot(
                np.array([robot_params.foot_spred, robot_params.foot_y, 0.]),
                robot_params.foot_size),
            com=PointMass(
                robot_params.com.copy()
            )
        )
        
        self.stance_foot: Foot = None
        self.swing_foot: Foot = None
        
    def on_tick(self):
        if self.state == WalkState.STAND:
            return self.run_standing()
        elif self.state == WalkState.DSP:
            return self.run_double_support()
        elif self.state == WalkState.SSP:
            return self.run_single_support()
        
    def start_standing(self):
        self.start_walking = False
        self.state = WalkState.STAND
        return self.run_standing()
    
    def run_standing(self):
        if self.start_walking:
            self.start_walking = False
            if self.next_footstep < len(self.footsteps):
                self.start_double_support()
    
    def start_double_support(self):
        if self.next_footstep % 2 == 1:
            self.stance_foot = self.stance.left_foot
            self.swing_foot = self.stance.right_foot
        else:
            self.stance_foot = self.stance.right_foot
            self.swing_foot = self.stance.left_foot
                    
        dsp_duration = self.dsp_duration
        if self.next_footstep == 2 or self.next_footstep == len(self.footsteps) - 1:
            dsp_duration *= 4
        
        self.swing_target = self.footsteps[self.next_footstep]
        self.rem_time = dsp_duration
        self.state = WalkState.DSP
        self.start_com_mpc_dsp()
        return self.run_double_support()
    
    def run_double_support(self):
        if self.rem_time <= 0.:
            return self.start_single_support()
        self.run_com_mpc()
        self.rem_time -= self.dt
    
    def start_single_support(self):
        if self.next_footstep % 2 == 1:
            self.swing_foot = self.stance.right_foot
            self.stance_foot = self.stance.left_foot
        else:
            self.swing_foot = self.stance.left_foot
            self.stance_foot = self.stance.right_foot
                    
        self.next_footstep += 1
        self.rem_time = self.ssp_duration
        self.state = WalkState.SSP
        self.start_swing_foot()
        self.start_com_mpc_ssp()
        self.run_single_support()
        
    def run_single_support(self):
        if self.rem_time <= 0:
            if self.next_footstep < len(self.footsteps):
                return self.start_double_support()
            else:
                return self.start_standing()
        self.run_swing_foot()
        self.run_com_mpc()
        self.rem_time -= self.dt
    
    
    def start_swing_foot(self):
        self.swing_start = self.swing_foot.position.copy()
        self.swing_interpolator = CubicHermiteInterpolation(
            start_pos=self.swing_start,
            end_pos=self.swing_target.position,
            duration=self.ssp_duration,
            n0=(0, 0, 1),
            n1=(0, 0, 1),
            takeoff_clearance=0.01,
            landing_clearance=0.01,
            s_takeoff=0.4,
            s_landing=0.6,
        )
    
    def run_swing_foot(self):
        new_pos = self.swing_interpolator.integrate(self.dt)
        self.swing_foot.position = new_pos
    
    def update_mpc(self, dsp_duration, ssp_duration):
        nb_preview_steps = 20
        T = 3 * self.dt
        nb_init_dsp_steps = int(round(dsp_duration / T))
        nb_init_ssp_steps = int(round(ssp_duration / T))
        nb_dsp_steps = int(round(self.dsp_duration / T))
        
        A = np.array([
            [1., T, T**2 / 2],
            [0., 1., T],
            [0., 0., 1.]
        ])
        B = np.array([
            [T**3 / 6],
            [T**2 / 2],
            [T]
        ])
        h = self.stance.com.position[2]
        g = 9.81
        zmp_from_state = np.array([1., 0., -h / g])
        C = np.array([zmp_from_state, -zmp_from_state])
        D = None
        e = [[], []]
        
        cur_vertices = self.stance_foot.get_scaled_contact_area(0.8)
        next_vertices = self.swing_foot.get_scaled_contact_area(0.8)
        
        for coord in [0, 1]:
            cur_max = max(v[coord] for v in cur_vertices)
            cur_min = min(v[coord] for v in cur_vertices)
            next_max = max(v[coord] for v in next_vertices)
            next_min = min(v[coord] for v in next_vertices)
            e[coord] = [
                np.array([1000., 1000.]) if i < nb_init_dsp_steps else
                np.array([cur_max, -cur_min]) if i - nb_init_dsp_steps <= nb_init_ssp_steps else
                np.array([1000., 1000.]) if i - nb_init_dsp_steps - nb_init_ssp_steps < nb_dsp_steps else
                np.array([next_max, -next_min])
                for i in range(nb_preview_steps)
            ]
        self.x_mpc = LinearPredictiveControl(
            A, B, C, D, e[0],
            x_init=np.array([self.stance.com.position[0], 
                             self.stance.com.velocity[0],
                             self.stance.com.acceleration[0]]),
            x_goal=np.array([self.swing_target.position[0], 0., 0.]),
            nb_steps=nb_preview_steps,
            wxt=1.,
            wu=0.01
        )
        self.y_mpc = LinearPredictiveControl(
            A, B, C, D, e[1],
            x_init=np.array([self.stance.com.position[1], 
                             self.stance.com.velocity[1],
                             self.stance.com.acceleration[1]]),
            x_goal=np.array([self.swing_target.position[1], 0., 0.]),
            nb_steps=nb_preview_steps,
            wxt=1.,
            wu=0.01
        )
        self.x_mpc.solve()
        self.y_mpc.solve()
        self.preview_time = 0
        
    
    def start_com_mpc_dsp(self):
        self.update_mpc(self.rem_time, self.ssp_duration)
    
    def start_com_mpc_ssp(self):
        self.update_mpc(0., self.rem_time)

    def run_com_mpc(self):
        # self.stance.com = np.array([
        #     0.,
        #     (self.swing_foot.position[1] + self.stance_foot.position[1]) / 2.0,
        #     self.stance.com.position[2]
        # ])
        if self.preview_time >= 3 * self.dt:
            if self.state == WalkState.DSP:
                self.update_mpc(self.rem_time, self.ssp_duration)
            elif self.state == WalkState.SSP:
                self.update_mpc(0., self.rem_time)
        com_jerk = np.array([
            self.x_mpc.U[0, 0],
            self.y_mpc.U[0, 0],
            0.
        ])
        self.stance.com.integrate_constant_jerk(com_jerk, self.dt)
        self.preview_time += self.dt