import pinocchio as pin
import qpsolvers
from classes import *
from generate_footsteps import generate_footsteps
from viberobotics.utils.utils import get_asset_path
from pink.tasks import ComTask, FrameTask
import numpy as np
import pink
from pink.solve_ik import solve_ik
from typing import List, Union
from fsm import WalkingFSM
from viser_vis import ViserVisualizer
import time
from loop_rate_limiters import RateLimiter
import mujoco, mujoco.viewer
from pathlib import Path
from viberobotics.motor.motor_controller_manager import MotorControllerManager
from viberobotics.configs.config import load_config


class Robot:
    def __init__(self):
        self.config = RobotConfig(
            xml_path=get_asset_path('mujoco/SundayA1_ankle_2dof_new/robot.xml'),
            left_foot_name='sole0120',
            right_foot_name='sole0120mir',
            foot_size=np.array([0.035, 0.09, 0.005]) * 2
        )
        self.walk_config = WalkConfig(
            ssp_duration=0.7,
            dsp_duration=0.07,
            step_length=0.04,
        )
        
        self.robot = pin.RobotWrapper.BuildFromMJCF(filename=self.config.xml_path, root_joint=None)
        self.model = self.robot.model
        self.nq = self.model.nq
    
        self.default_q = np.zeros(self.nq)
        self.default_q[0:7] = np.array([0, 0, 0., 0, 0, 0, 1])
        self.q = self.default_q.copy()
    
        pin.forwardKinematics(self.model, self.robot.data, self.default_q)
        pin.updateFramePlacements(self.model, self.robot.data)
        
        self.params = self.get_params()
        
        self.pink_configuration = pink.Configuration(self.robot.model, self.robot.data, self.default_q)
        self.left_foot_task = FrameTask(
            self.config.left_foot_name,
            position_cost=4.0,
            orientation_cost=5.0,
        )
        self.right_foot_task = FrameTask(
            self.config.right_foot_name,
            position_cost=4.0,
            orientation_cost=5.0,
        )
        self.com_task = ComTask(
            cost=100.0,
        )
        self.ik_tasks: List[Union[FrameTask, ComTask]] = [
            self.left_foot_task,
            self.right_foot_task,
            self.com_task
        ]
        for task in self.ik_tasks:
            task.set_target_from_configuration(self.pink_configuration)
        self.solver = qpsolvers.available_solvers[0]
        if "proxqp" in qpsolvers.available_solvers:
            self.solver = "proxqp"
            
        self.footsteps = generate_footsteps(1.0, self.walk_config.step_length, self.params.foot_spred, initial_y=0.05)
        self.fsm = WalkingFSM(
            self.walk_config.ssp_duration,
            self.walk_config.dsp_duration,
            self.footsteps,
            self.config,
            self.params
        )
    
    def get_params(self):
        pin.centerOfMass(self.model, self.robot.data, self.default_q)
        com = np.asarray(self.robot.data.com[0])
    
        left_foot_frame_id = self.model.getFrameId(self.config.left_foot_name)
        right_foot_frame_id = self.model.getFrameId(self.config.right_foot_name)
        left_foot_frame_pos = self.robot.data.oMf[left_foot_frame_id].translation.copy()
        right_foot_frame_pos = self.robot.data.oMf[right_foot_frame_id].translation.copy()
        
        pin.updateGeometryPlacements(self.model, self.robot.data, self.robot.collision_model, self.robot.collision_data)
    
        left_foot_pos = self.robot.collision_data.oMg[1].translation.copy()
        right_foot_pos = self.robot.collision_data.oMg[0].translation.copy()
        
        foot_spred = np.linalg.norm(left_foot_pos[0] - right_foot_pos[0]) / 2.0
        return RobotParams(
            com=com,
            foot_spred=foot_spred,
            foot_size=self.config.foot_size,
            foot_y=(left_foot_pos[1] + right_foot_pos[1]) / 2.0,
            left_foot_offset=left_foot_pos - left_foot_frame_pos,
            right_foot_offset=right_foot_pos - right_foot_frame_pos,
        )
        
    def ik(self,
       ik_target: IKTarget,
       max_iters: int = 500,
       tol: float = 1e-3,
       damping: float = 3e-2,
       step: float = 0.01) -> np.ndarray:

        offset = np.array([ik_target.com_pos[0], ik_target.com_pos[1], 0.])
        ik_target = IKTarget(
            left_foot_pos=ik_target.left_foot_pos - offset,
            right_foot_pos=ik_target.right_foot_pos - offset,
            com_pos=ik_target.com_pos - offset
        )

        left_geom_name = "sole0120_0"
        right_geom_name = "sole0120mir_0"

        model = self.robot.model
        gmodel = self.robot.collision_model
        geoms = gmodel.geometryObjects
    
        def _ensure_collider_frame():
            name_to_gid = {g.name: i for i, g in enumerate(geoms)}

            def add_for_geom(geom_name: str, foot_body_frame_name: str, new_frame_name: str):
                if model.existFrame(new_frame_name):
                    return

                if geom_name not in name_to_gid:
                    raise ValueError(f"Collision geom '{geom_name}' not found in collision_model.geometryObjects")

                g = geoms[name_to_gid[geom_name]]

                parent_joint = int(g.parentJoint)

                parent_frame = getattr(g, "parentFrame", None)
                if parent_frame is None or parent_frame < 0 or parent_frame >= len(model.frames):
                    parent_frame = model.getFrameId(foot_body_frame_name)
                    if parent_frame == len(model.frames):
                        raise ValueError(f"Fallback parent frame '{foot_body_frame_name}' not found in model.frames")
                parent_frame = int(parent_frame)

                placement = g.placement
                f = pin.Frame(new_frame_name, parent_joint, parent_frame, placement, pin.FrameType.FIXED_JOINT)
                model.addFrame(f)

            add_for_geom(left_geom_name, self.config.left_foot_name, f"{left_geom_name}_frame")
            add_for_geom(right_geom_name, self.config.right_foot_name, f"{right_geom_name}_frame")

        _ensure_collider_frame()

        # IMPORTANT: re-create data AFTER adding frames
        data = model.createData()
        self.robot.data = data

        q = pin.neutral(model).copy()

        left_fid = model.getFrameId(f"{left_geom_name}_frame")
        right_fid = model.getFrameId(f"{right_geom_name}_frame")
        if left_fid == len(model.frames) or right_fid == len(model.frames):
            raise ValueError(f"Frame not found: {left_geom_name}_frame or {right_geom_name}_frame")

        # --- base frame id for relative errors ---
        # Use whichever you actually have:
        # base_fid = model.getFrameId(BASE_NAME)
        base_fid = model.getFrameId('base')
        if base_fid == len(model.frames):
            raise ValueError("Base frame not found (BASE_NAME / self.config.base_name).")

        # --- flat reference orientations (as you had) ---
        data_ref = model.createData()
        q_flat = pin.neutral(model).copy()
        pin.forwardKinematics(model, data_ref, q_flat)
        pin.updateFramePlacements(model, data_ref)
        R_L_flat = data_ref.oMf[left_fid].rotation.copy()
        R_R_flat = data_ref.oMf[right_fid].rotation.copy()

        for _ in range(max_iters):
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacements(model, data)
            pin.centerOfMass(model, data, q)

            oMfL = data.oMf[left_fid]
            oMfR = data.oMf[right_fid]
            oMfB = data.oMf[base_fid]   # world pose of base frame

            pL = oMfL.translation
            pR = oMfR.translation
            pC = data.com[0]

            # --- FIX: compute errors in base frame (translation-invariant) ---
            Rwb = oMfB.rotation
            Rbw = Rwb.T

            eL_pos_w = (ik_target.left_foot_pos - pL).reshape(3)
            eR_pos_w = (ik_target.right_foot_pos - pR).reshape(3)
            eC_w     = (ik_target.com_pos - pC).reshape(3)

            eL_pos = Rbw @ eL_pos_w
            eR_pos = Rbw @ eR_pos_w
            eC     = Rbw @ eC_w

            # orientation errors (still in rotation space; OK to keep as-is)
            R_err_L = R_L_flat.T @ oMfL.rotation
            R_err_R = R_R_flat.T @ oMfR.rotation
            eL_ori = pin.log3(R_err_L)
            eR_ori = pin.log3(R_err_R)

            w_ori = 1.0
            e = np.concatenate([eL_pos, w_ori * eL_ori, eR_pos, w_ori * eR_ori, eC], axis=0)

            if np.linalg.norm(e) < tol:
                break

            JL6_w = pin.computeFrameJacobian(model, data, q, left_fid, pin.ReferenceFrame.WORLD)
            JR6_w = pin.computeFrameJacobian(model, data, q, right_fid, pin.ReferenceFrame.WORLD)

            # --- FIX: rotate translational Jacobian blocks into base frame to match eL_pos/eR_pos/eC ---
            JL_pos = Rbw @ JL6_w[:3, :]
            JR_pos = Rbw @ JR6_w[:3, :]

            # orientation Jacobians: keep world (matches log3 error you used)
            JL_ori = JL6_w[3:6, :]
            JR_ori = JR6_w[3:6, :]

            # CoM jacobian is in world; rotate to base to match eC
            JC_w = pin.jacobianCenterOfMass(model, data, q)
            JC = Rbw @ JC_w

            J = np.vstack([
                JL_pos,
                w_ori * JL_ori,
                JR_pos,
                w_ori * JR_ori,
                JC
            ])  # (15, nv)

            A = (J @ J.T) + (damping ** 2) * np.eye(J.shape[0])
            dq = J.T @ np.linalg.solve(A, e)
            q = pin.integrate(model, q, step * dq)

        return q + np.concatenate([offset, np.zeros(self.nq - 3)])
        
    
    def visualize(self):
        visualizer = ViserVisualizer(get_asset_path('urdf/SundayA1_ankle_2dof_new/robot.urdf'))
        running = False
        start_button = visualizer.server.gui.add_button('start')
        @start_button.on_click
        def _(_) -> None:
            nonlocal running
            running = not running
            if not running:
                start_button.label = 'start'
            else:
                start_button.label = 'stop'
                self.fsm.start_walking = True
        
        t_slider = visualizer.server.gui.add_number('t', 0., min=0., max=100., step=0.01, disabled=True)
        
        left_foot_marker = visualizer.server.scene.add_icosphere('/left_foot_marker', radius=0.02, color=(255, 0, 0))
        right_foot_marker = visualizer.server.scene.add_icosphere('/right_foot_marker', radius=0.02, color=(0, 255, 0))
        com_marker = visualizer.server.scene.add_icosphere('/com_marker', radius=0.02, color=(0, 0, 255))
        
        dt = 0.03
        rate_limiter = RateLimiter(frequency=1/dt, warn=False)
        while True:
            self.fsm.on_tick()
            t_slider.value += dt
            
            left_foot_marker.position = self.fsm.stance.left_foot.position
            right_foot_marker.position = self.fsm.stance.right_foot.position
            com_marker.position = self.fsm.stance.com.position

            
            self.q = self.ik(IKTarget(
                left_foot_pos=left_foot_marker.position,
                right_foot_pos=right_foot_marker.position,
                com_pos=com_marker.position,
            ))
            
            if self.fsm.stance_foot is not None:
                contact_verts = self.fsm.stance_foot.get_scaled_contact_area(0.8)
                points = np.array([
                    [contact_verts[0] - np.array([0, 0, 0.01]), contact_verts[1] - np.array([0, 0, 0.01])],
                    [contact_verts[1] - np.array([0, 0, 0.01]), contact_verts[2] - np.array([0, 0, 0.01])],
                    [contact_verts[2] - np.array([0, 0, 0.01]), contact_verts[3] - np.array([0, 0, 0.01])],
                    [contact_verts[3] - np.array([0, 0, 0.01]), contact_verts[0] - np.array([0, 0, 0.01])],
                ])
                visualizer.server.scene.add_line_segments(
                    '/contact_area',
                    points=points,
                    colors=np.array([0, 1, 1]),
                    line_width=3.0,
                )
            
            visualizer.set_state(self.q)
            rate_limiter.sleep()
    
    def simulate(self):
        model = mujoco.MjModel.from_xml_path((Path(self.config.xml_path).parent / 'scene.xml').as_posix())
        data = mujoco.MjData(model)
        print(model.opt.timestep)
        
        self.fsm.start_walking = True
        dt = 0.03
        rate_limiter = RateLimiter(frequency=1/dt, warn=False)
        kp = 200
        kd = 10
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while True:
                self.fsm.on_tick()
                
                self.q = self.ik(IKTarget(
                    left_foot_pos=self.fsm.stance.left_foot.position,
                    right_foot_pos=self.fsm.stance.right_foot.position,
                    com_pos=self.fsm.stance.com.position,
                ))
                for _ in range(int(dt / model.opt.timestep)):
                    cur_q = data.qpos[7:].copy()
                    cur_qd = data.qvel[6:].copy()
                    tau = kp * (self.q[7:] - cur_q) - kd * cur_qd
                    data.ctrl[:] = np.clip(tau, -4., 4.)
                    mujoco.mj_step(model, data)
                    viewer.sync()
    
    def deploy(self):
        cfg = load_config('sundaya1_real_config_half_2dof.yaml')
        motor_manager = MotorControllerManager(
            cfg.real_config.n_motors,
            cfg.real_config.motor_controllers,
            cfg.real_config.calibration_file,
            mode=0
        )
        motor_manager.set_positions(cfg.default_qpos, 0, 50)
        
        input('start>')
        
        try:
            dt = 0.06
            rate_limiter = RateLimiter(frequency=1 / dt, warn=True)
            self.fsm.start_walking = True
            while True:
                self.fsm.on_tick()

                self.q = self.ik(IKTarget(
                    left_foot_pos=self.fsm.stance.left_foot.position,
                    right_foot_pos=self.fsm.stance.right_foot.position,
                    com_pos=self.fsm.stance.com.position,
                ))
                
                # rate_limiter_inner = RateLimiter(frequency=1 / 0.002, warn=True)
                # for _ in range(int(dt / 0.002)):
                #     duty = 32 * (self.q[7:] - motor_manager.get_state()[0]) - 1.5 * motor_manager.get_state()[1]
                #     motor_manager.set_duty(duty * 200)
                #     rate_limiter_inner.sleep()
                
                motor_manager.set_positions(self.q[7:], 0, 50)
                rate_limiter.sleep()
        except KeyboardInterrupt:
            motor_manager.disable_torque()
        
        
        
    
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='view', choices=['view', 'simulate', 'deploy'], help='Mode: view, simulate, deploy')
    args = parser.parse_args()
    
    robot = Robot()
    if args.mode == 'view':
        robot.visualize()
    elif args.mode == 'simulate':
        robot.simulate()
    elif args.mode == 'deploy':
        robot.deploy()