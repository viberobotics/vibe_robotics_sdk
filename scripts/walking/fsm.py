from classes import *
from interp import *
from typing import List
from mpc import LinearPredictiveControl
import numpy as np
from generate_footsteps import FootstepGenerator
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import time

class WalkingFSM:
    def __init__(self, 
                 walk_config: WalkConfig,
                 footsteps: List[Footstep],
                 robot_config: RobotConfig,
                 robot_params: RobotParams):
        self.ssp_duration = walk_config.ssp_duration
        self.dsp_duration = walk_config.dsp_duration
        self.footsteps = footsteps
        
        self.state: WalkState = WalkState.STAND
        
        self.start_walking = False
        self.next_footstep = 2
        
        self.dt = 0.03
        self.mpc_interval = 3 * self.dt
        self.stance = Stance(
            left_foot=Foot(
                FootType.LEFT,
                np.array([-robot_params.foot_spred, robot_params.foot_y, 0.]),
                robot_params.foot_size),
            right_foot=Foot(
                FootType.RIGHT,
                np.array([robot_params.foot_spred, robot_params.foot_y, 0.]),
                robot_params.foot_size),
            com=PointMass(
                robot_params.com.copy()
            )
        )
        
        self.stance_foot: Foot = None
        self.swing_foot: Foot = None
        
        self.footstep_generator = FootstepGenerator(
            step_length=walk_config.step_length,
            foot_spread=robot_params.foot_spred,
            initial_y=robot_params.foot_y,
            steering_strength=np.deg2rad(5.)
        )
        self.cmd = WalkCommand.STRAIGHT
        self.last_cmd = WalkCommand.STRAIGHT
        self.tick = 0
        self.last_change_cmd_tick = 0
    
    def set_cmd(self, cmd: WalkCommand):
        if self.cmd == WalkCommand.STOP and cmd != WalkCommand.STOP:
            self.start_walking = True
        self.cmd = cmd
        if self.cmd != self.last_cmd:
            self.last_cmd = self.cmd
            self.last_change_cmd_tick = self.tick
    
    def on_tick(self):
        self.tick += 1
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
            self.start_double_support()
    
    def start_double_support(self):
        if self.next_footstep % 2 == 1:
            self.stance_foot = self.stance.left_foot
            self.swing_foot = self.stance.right_foot
        else:
            self.stance_foot = self.stance.right_foot
            self.swing_foot = self.stance.left_foot
                    
        dsp_duration = self.dsp_duration
        if self.next_footstep == 2:
            dsp_duration *= 4
        
        # self.swing_target = self.footsteps[self.next_footstep]
        self.swing_target = self.footstep_generator.get_next_footstep(
            self.cmd,
            self.stance_foot.side
        )
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
            if self.cmd == WalkCommand.STOP:
                return self.start_standing()
            return self.start_double_support()
        self.run_swing_foot()
        self.run_com_mpc()
        self.rem_time -= self.dt
    
    
    def start_swing_foot(self):
        self.swing_start = self.swing_foot
        self.swing_interpolator = CubicHermiteInterpolation(
            start_pose=self.swing_start,
            end_pose=self.swing_target,
            duration=self.ssp_duration,
            n0=(0, 0, 1),
            n1=(0, 0, 1),
            takeoff_clearance=0.03,
            landing_clearance=0.03,
            s_takeoff=0.25,
            s_landing=0.75,
        )
    
    def run_swing_foot(self):
        new_pose = self.swing_interpolator.integrate(self.dt)
        self.swing_foot.T = new_pose.T
    
    def update_mpc(self, dsp_duration, ssp_duration):
        start_time = time.time()
        nb_preview_steps = 10

        # Planning timestep for the jerk-integrator model
        T = self.mpc_interval
        nb_init_dsp_steps = int(round(dsp_duration / T))
        nb_init_ssp_steps = int(round(ssp_duration / T))
        nb_dsp_steps = int(round(self.dsp_duration / T))

        # --- Discrete jerk integrator (same as before) ---
        A = np.array([
            [1., T, T**2 / 2.],
            [0., 1., T],
            [0., 0., 1.]
        ])
        B = np.array([
            [T**3 / 6.],
            [T**2 / 2.],
            [T]
        ])

        # --- Heading-aligned frame ---
        #
        # IMPORTANT: this theta should be your "navigation heading" (the thing you steer),
        # NOT the pelvis/COM yaw that sways.
        #
        # If you have a FootstepGenerator, store its heading as self.nav_theta and use it here.
        #
        # Fallbacks:
        # - if swing_target has yaw: use it
        # - else: use 0
        theta = self.footstep_generator.ref_theta

        c, s = np.cos(theta), np.sin(theta)

        # World (x right, y forward) -> local (f forward, l left/right) mapping:
        #
        # forward unit in world = [sin(theta), cos(theta)]
        # right unit   in world = [cos(theta), -sin(theta)]
        #
        # local coordinates:
        #   f = dot([x,y], fwd_unit)
        #   l = dot([x,y], right_unit)
        #
        fwd = np.array([s, c])
        right = np.array([c, -s])

        def world_xy_to_fl(xy):
            xy = np.asarray(xy, dtype=float)
            return np.array([xy @ fwd, xy @ right], dtype=float)

        def world_xyvel_to_fld(vel_xy):
            vel_xy = np.asarray(vel_xy, dtype=float)
            return np.array([vel_xy @ fwd, vel_xy @ right], dtype=float)

        # We'll need this later in run_com_mpc() to rotate jerk back to world:
        # world = f * fwd + l * right
        self._fl_to_world = np.stack([fwd, right], axis=1)  # shape (2,2): columns are [fwd, right]

        # --- ZMP constraint in 1D form: zmp = [1, 0, -h/g] x_axis_state ---
        h = float(self.stance.com.position[2])
        g = 9.81
        zmp_from_state = np.array([1., 0., -h / g])
        C = np.array([zmp_from_state, -zmp_from_state])  # (2,3)
        D = None

        # --- Support bounds (now computed in LOCAL frame) ---
        #
        # We still use simple min/max bounds, but crucially in the (f,l) frame.
        # That makes the constraints "rotate with heading" so lateral sway doesn't vanish when you steer.
        cur_vertices_w = self.stance_foot.get_scaled_contact_area(0.8)
        next_vertices_w = self.swing_foot.get_scaled_contact_area(0.8)

        # Convert vertex lists into local (f,l) coords
        cur_vertices_fl = [world_xy_to_fl(v[:2]) for v in cur_vertices_w]
        next_vertices_fl = [world_xy_to_fl(v[:2]) for v in next_vertices_w]

        e = [[], []]  # e[0] for forward (f), e[1] for lateral (l)

        for coord in [0, 1]:  # 0=f, 1=l
            cur_max = max(v[coord] for v in cur_vertices_fl)
            cur_min = min(v[coord] for v in cur_vertices_fl)
            next_max = max(v[coord] for v in next_vertices_fl)
            next_min = min(v[coord] for v in next_vertices_fl)

            # Same time-scheduling logic as your code, but now the bounds live in (f,l)
            e[coord] = [
                np.array([1000., 1000.]) if i < nb_init_dsp_steps else
                np.array([cur_max, -cur_min]) if (i - nb_init_dsp_steps) <= nb_init_ssp_steps else
                np.array([1000., 1000.]) if (i - nb_init_dsp_steps - nb_init_ssp_steps) < nb_dsp_steps else
                np.array([next_max, -next_min])
                for i in range(nb_preview_steps)
            ]

        # --- Initial state / goal in LOCAL frame ---
        com_xy = self.stance.com.position[:2]
        comd_xy = self.stance.com.velocity[:2]
        comdd_xy = self.stance.com.acceleration[:2]

        com_fl = world_xy_to_fl(com_xy)
        comd_fl = world_xyvel_to_fld(comd_xy)
        comdd_fl = world_xyvel_to_fld(comdd_xy)
        
        omega = np.sqrt(9.81 / h)   # h = com height you already use
        S_dcm = np.array([[1.0, 1.0 / omega, 0.0]])
        

        # Goal: use your swing_target position but expressed in (f,l)
        # (You may later want a different goal, e.g. mid-support or capture-point ref,
        # but this keeps your structure unchanged.)
        goal_fl = world_xy_to_fl(self.swing_target.position[:2])

        fwd_adjust = 0.05 if self.cmd == WalkCommand.STRAIGHT else 0.02
        if self.tick - self.last_change_cmd_tick < 30:  # for the first 10 ticks after a command change, add a forward bias to encourage responsiveness
            if self.cmd == WalkCommand.RIGHT:
                if self.last_cmd == WalkCommand.STRAIGHT:
                    fwd_adjust = 0.09
                elif self.last_cmd == WalkCommand.LEFT:
                    fwd_adjust = 0.125
                elif self.last_cmd == WalkCommand.STOP:
                    fwd_adjust = 0.05
            elif self.cmd == WalkCommand.LEFT:
                if self.last_cmd == WalkCommand.STRAIGHT:
                    fwd_adjust = 0.15
                elif self.last_cmd == WalkCommand.RIGHT:
                    fwd_adjust = 0.0
        
        # MPC in forward axis
        self.f_mpc = LinearPredictiveControl(
            A, B, C, D, e[0],
            x_init=np.array([com_fl[0], comd_fl[0], comdd_fl[0]]),
            x_goal=np.array([goal_fl[0] + fwd_adjust, 0., 0.]),
            nb_steps=nb_preview_steps,
            wxt=2.,
            wu=0.01,
        )

        # MPC in lateral axis
        self.l_mpc = LinearPredictiveControl(
            A, B, C, D, e[1],
            x_init=np.array([com_fl[1], comd_fl[1], comdd_fl[1]]),
            x_goal=np.array([goal_fl[1] * 0.2, 0., 0.]),
            nb_steps=nb_preview_steps,
            wxt=1.,
            wu=0.01,
            wxc=3.
        )

        self.f_mpc.solve()
        self.l_mpc.solve()
        self.preview_time = 0.0
        # print('MPC solve time: {:.3f}s'.format(time.time() - start_time))
        
    def update_mpc_old(self, dsp_duration, ssp_duration):
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
        if self.preview_time >= self.mpc_interval:
            if self.state == WalkState.DSP:
                self.update_mpc(self.rem_time, self.ssp_duration)
            elif self.state == WalkState.SSP:
                self.update_mpc(0., self.rem_time)
        j_fl = np.array([self.f_mpc.U[0, 0], self.l_mpc.U[0, 0]])   # [j_f, j_l]
        j_xy = self._fl_to_world @ j_fl                              # 2x2 @ 2
        com_jerk = np.array([j_xy[0], j_xy[1], 0.0])
        self.stance.com.integrate_constant_jerk(com_jerk, self.dt)
        self.preview_time += self.dt