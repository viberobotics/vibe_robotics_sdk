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
from scipy.spatial.transform import Rotation as R
import pygame
import viser.uplot

pygame.init()

def _skew(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(3)
    return np.array([[0.0, -v[2],  v[1]],
                     [v[2],  0.0, -v[0]],
                     [-v[1], v[0], 0.0]], dtype=float)

def rot_from_two_unit_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Return R such that R @ a = b (a,b are 3D, can be non-unit; we normalize).
    Handles parallel and antiparallel cases.
    """
    a = np.asarray(a, dtype=float).reshape(3)
    b = np.asarray(b, dtype=float).reshape(3)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return np.eye(3)
    a = a / na
    b = b / nb

    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = np.linalg.norm(v)

    if s < 1e-12:
        # parallel or anti-parallel
        if c > 0.0:
            return np.eye(3)
        # 180 deg: choose any axis orthogonal to a
        axis = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        axis = axis - a * float(np.dot(a, axis))
        axis = axis / max(np.linalg.norm(axis), 1e-12)
        K = _skew(axis)
        # Rodrigues for pi: R = I + 2 K^2
        return np.eye(3) + 2.0 * (K @ K)

    axis = v / s
    K = _skew(axis)
    # Rodrigues: R = I + sinθ K + (1-cosθ)K^2, where cosθ=c, sinθ=s
    return np.eye(3) + s * K + (1.0 - c) * (K @ K)

def _pick_local_forward_axis(R_flat: np.ndarray) -> np.ndarray:
    """
    Choose which local axis of the foot frame best corresponds to "forward"
    by checking which of ±X, ±Y, ±Z is closest to world +Y under R_flat.
    """
    world_fwd = np.array([0.0, 1.0, 0.0], dtype=float)
    cands = [
        np.array([ 1.0, 0.0, 0.0]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([ 0.0, 1.0, 0.0]),
        np.array([ 0.0,-1.0, 0.0]),
        np.array([ 0.0, 0.0, 1.0]),
        np.array([ 0.0, 0.0,-1.0]),
    ]
    best = cands[0]
    best_score = -1e9
    for a in cands:
        v = R_flat @ a
        score = float(np.dot(v / max(np.linalg.norm(v), 1e-12), world_fwd))
        if score > best_score:
            best_score = score
            best = a
    return best
def _roty(theta: float) -> np.ndarray:
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.array([[ c, 0.0,  s],
                     [0.0, 1.0, 0.0],
                     [-s, 0.0,  c]], dtype=float)
def _rotz(theta: float) -> np.ndarray:
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.array([[ c, -s, 0.0],
                     [ s,  c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=float)

def _yaw_align_to_plus_y(R_flat: np.ndarray, local_fwd: np.ndarray) -> np.ndarray:
    """
    Returns R_des = Rz(delta_yaw) @ R_flat such that
    the projected forward direction aligns with world +Y.
    """
    world_fwd_des = np.array([0.0, 1.0, 0.0], dtype=float)

    fwd_w = R_flat @ local_fwd
    fwd_xy = np.array([fwd_w[0], fwd_w[1]], dtype=float)
    if np.linalg.norm(fwd_xy) < 1e-9:
        # degenerate: no horizontal component; do nothing
        return R_flat.copy()
    fwd_xy /= np.linalg.norm(fwd_xy)

    des_xy = world_fwd_des[:2]  # [0,1]
    # current yaw: atan2(y, x), desired yaw for +Y is pi/2
    yaw_cur = float(np.arctan2(fwd_xy[1], fwd_xy[0]))
    yaw_des = float(np.arctan2(des_xy[1], des_xy[0]))  # pi/2
    delta = yaw_des - yaw_cur

    return _rotz(delta) @ R_flat

def Rz(theta: float) -> np.ndarray:
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.array([[ c, -s, 0.0],
                    [ s,  c, 0.0],
                    [0.0, 0.0, 1.0]], dtype=float)
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
            step_length=0.03,
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
            
        # self.footsteps = generate_footsteps(1.0, self.walk_config.step_length, self.params.foot_spred, initial_y=0.05)
        self.fsm = WalkingFSM(
            self.walk_config,
            None,
            self.config,
            self.params
        )
        
        
        pygame.joystick.init()
        assert pygame.joystick.get_count() > 0
        self.joystick = pygame.joystick.Joystick(0)
        self.cmd = np.zeros(3)
        
    
    def get_params(self):
        pin.centerOfMass(self.model, self.robot.data, self.default_q)
        com = np.asarray(self.robot.data.com[0])
    
        left_foot_frame_id = self.model.getFrameId(self.config.left_foot_name)
        right_foot_frame_id = self.model.getFrameId(self.config.right_foot_name)
        left_foot_frame_pos = self.robot.data.oMf[left_foot_frame_id].translation.copy()
        right_foot_frame_pos = self.robot.data.oMf[right_foot_frame_id].translation.copy()
        
        pin.updateGeometryPlacements(self.model, self.robot.data, self.robot.collision_model, self.robot.collision_data)
        model = self.robot.model
        gmodel = self.robot.collision_model
        geoms = gmodel.geometryObjects
        name_to_gid = {g.name: i for i, g in enumerate(geoms)}
        print(name_to_gid)
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
       max_iters: int = 300,
       tol: float = 2e-3,
       damping: float = 3e-2,
       step: float = 0.05,
       update_heading: bool=False) -> np.ndarray:

        offset = np.array([ik_target.com_pos[0], ik_target.com_pos[1], 0.])
        ik_target = IKTarget(
            left_foot_pose=ik_target.left_foot_pose.translated(-offset),
            right_foot_pose=ik_target.right_foot_pose.translated(-offset),
            com_pos=ik_target.com_pos - offset,
            heading=ik_target.heading
        )

        left_geom_name = self.config.left_foot_name + "_0"
        right_geom_name = self.config.right_foot_name + "_0"

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
        q[3:7] = self.q[3:7].copy()
        # q = self.q.copy()
        

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

            eL_pos_w = (ik_target.left_foot_pose.position - pL).reshape(3)
            eR_pos_w = (ik_target.right_foot_pose.position - pR).reshape(3)
            eC_w     = (ik_target.com_pos - pC).reshape(3)

            eL_pos = Rbw @ eL_pos_w
            eR_pos = Rbw @ eR_pos_w
            eC     = Rbw @ eC_w

            # orientation errors (still in rotation space; OK to keep as-is)
            R_err_L = R_L_flat.T @ oMfL.rotation
            R_err_R = R_R_flat.T @ oMfR.rotation
            
            Rcmd_L = ik_target.left_foot_pose.rotation.as_matrix()   # what you pass in (don’t change it)
            Rcmd_R = ik_target.right_foot_pose.rotation.as_matrix()

            # Interpret Rcmd as "relative to flat foot frame"
            Rdes_L_w = Rcmd_L @ R_L_flat
            Rdes_R_w = Rcmd_R @ R_R_flat
            eL_ori = pin.log3(Rdes_L_w @ oMfL.rotation.T)
            eR_ori = pin.log3(Rdes_R_w @ oMfR.rotation.T)


            # base orientation error (world): log(Rdes * Rcur^T)+
                # current base rotation
            Rcur_B = oMfB.rotation
            Rdes_B_w = Rz(-ik_target.heading)
            Rerr = Rdes_B_w @ Rcur_B.T
            yaw_err = float(np.arctan2(Rerr[1,0], Rerr[0,0]))  # yaw error about +Z
            eC_ori = np.array([yaw_err])

            w_ori = 1.0
            w_head = 1.0
            e = np.concatenate([eL_pos, w_ori*eL_ori, eR_pos, w_ori*eR_ori, eC, w_head*eC_ori], axis=0)
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
            
            JB6_w = pin.computeFrameJacobian(model, data, q, base_fid, pin.ReferenceFrame.WORLD)
            JB_yaw = JB6_w[5:6, :]

            # CoM jacobian is in world; rotate to base to match eC
            JC_w = pin.jacobianCenterOfMass(model, data, q)
            JC = Rbw @ JC_w
            J = np.vstack([
                JL_pos,
                w_ori * JL_ori,
                JR_pos,
                w_ori * JR_ori,
                JC,
                w_head * JB_yaw
            ])

            A = (J @ J.T) + (damping ** 2) * np.eye(J.shape[0])
            dq = J.T @ np.linalg.solve(A, e)
            q = pin.integrate(model, q, step * dq)

        success = np.linalg.norm(e) < tol
        return q + np.concatenate([offset, np.zeros(self.nq - 3)]), success
    
    def get_joystick_cmd(self):
        for event in pygame.event.get():
            pass
        self.cmd[0] = -self.joystick.get_axis(1)  # forward/backward
        self.cmd[1] = -self.joystick.get_axis(0)   # left/right
        self.cmd[2] = self.cmd[2] * 0.8 + 0.2 * self.joystick.get_axis(3)  # yaw
        return self.cmd
    
    def _get_targets(self):
        com_pos_nominal = self.fsm.stance.com.position
        theta = self.fsm.footstep_generator.ref_theta
        fwd = np.array([np.sin(theta), np.cos(theta)])     # world XY forward
        lat = np.array([np.cos(theta), -np.sin(theta)])    # world XY lateral/right
        center_xy = 0.5 * (
            self.fsm.stance.left_foot.position[:2]
            + self.fsm.stance.right_foot.position[:2]
        )
        d_xy = com_pos_nominal[:2] - center_xy
        d_f = d_xy @ fwd
        # d_f = np.clip(d_f, -0.01, 0.01)
        d_l = d_xy @ lat
        lateral_scale = 0.8 if np.abs(self.fsm.cmd[0]) < 0.1 or np.abs(self.cmd[1]) > 0.1 else 0.6
        forward_scale = 1.
        self.d_f = d_f
        com_xy = center_xy + (forward_scale * d_f) * fwd + (lateral_scale * d_l) * lat
        com_pos = np.array([com_xy[0], com_xy[1], com_pos_nominal[2]])
        return IKTarget(
            left_foot_pose=self.fsm.stance.left_foot,
            right_foot_pose=self.fsm.stance.right_foot,
            com_pos=com_pos,
            heading=self.fsm.footstep_generator.ref_theta
        )
        
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
        
        success_indicator = visualizer.server.gui.add_text('ik_success', 'IK solve success: N/A')
        
        left_foot_marker = visualizer.server.scene.add_icosphere('/left_foot_marker', radius=0.02, color=(255, 0, 0))
        right_foot_marker = visualizer.server.scene.add_icosphere('/right_foot_marker', radius=0.02, color=(0, 255, 0))
        com_marker = visualizer.server.scene.add_icosphere('/com_marker', radius=0.02, color=(0, 0, 255))
        
        dt = 0.03
        rate_limiter = RateLimiter(frequency=1/dt, warn=False)
        line_segments = []
        com_line_segments = []
        com_pos_list = []
        heading_update = 60
        heading_counter = 0
        
        t = [time.time()]
        d_f = [0]
        plot = visualizer.server.gui.add_uplot(
            data=(np.array(t), np.array(d_f)),
            series=(
                viser.uplot.Series(label='time'),
                viser.uplot.Series(label='d_f', stroke='blue', width=2),
            ),
            title='d_f over time',
            scales={
                'x': viser.uplot.Scale(time=True, auto=True),
                'y': viser.uplot.Scale(time=False, auto=True),
            },
            aspect=2.
        )
        
        while True:
            self.fsm.set_cmd(self.get_joystick_cmd())
            self.fsm.on_tick()
            t_slider.value += dt
            
            
            
            # Scale COM only along heading-lateral direction, not world X.
            # Use mid-foot as local origin so turning/global translation does not distort it.
            ik_target = self._get_targets()
            
            
            t.append(time.time())
            d_f.append(self.d_f)
            t = t[-200:]
            d_f = d_f[-200:]
            plot.data = (np.array(t), np.array(d_f))

            
            left_foot_marker.position = ik_target.left_foot_pose.position
            right_foot_marker.position = ik_target.right_foot_pose.position
            com_marker.position = ik_target.com_pos
            start_time = time.time()
            self.q, success = self.ik(ik_target, update_heading=(heading_counter % heading_update == 0), tol=0.01)
            
            success_indicator.label = f'IK solve success: {success}'
            # print(f"IK solve time: {time.time() - start_time:.3f}s")
            heading_counter += 1
            
            com_pos_list.append(ik_target.com_pos)
            com_pos_list = com_pos_list[-100:]
            if len(com_pos_list) > 2:
                com_line_segments.append([com_pos_list[-2], com_pos_list[-1]])
                visualizer.server.scene.add_line_segments(
                    '/com_trail',
                    points=np.array(com_line_segments),
                    colors=np.array([0, 0, 255]),
                    line_width=2.0,
                )
            
            
            if self.fsm.stance_foot is not None:
                contact_verts = self.fsm.stance_foot.get_scaled_contact_area(0.8)
                points = np.array([
                    [contact_verts[0] - np.array([0, 0, 0.01]), contact_verts[1] - np.array([0, 0, 0.01])],
                    [contact_verts[1] - np.array([0, 0, 0.01]), contact_verts[2] - np.array([0, 0, 0.01])],
                    [contact_verts[2] - np.array([0, 0, 0.01]), contact_verts[3] - np.array([0, 0, 0.01])],
                    [contact_verts[3] - np.array([0, 0, 0.01]), contact_verts[0] - np.array([0, 0, 0.01])],
                ])
                line_segments.append(points)
                visualizer.server.scene.add_line_segments(
                    '/contact_area',
                    points=np.concatenate(line_segments, axis=0),
                    colors=np.array([0, 0, 0]),
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
        self.fsm.set_cmd(WalkCommand.LEFT)
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while True:
                self.fsm.on_tick()
                
                self.q = self.ik(IKTarget(
                    left_foot_pose=self.fsm.stance.left_foot,
                    right_foot_pose=self.fsm.stance.right_foot,
                    com_pos=self.fsm.stance.com.position,
                    heading=self.fsm.footstep_generator.ref_theta
                ))
                for _ in range(int(dt / model.opt.timestep)):
                    cur_q = data.qpos[7:].copy()
                    cur_qd = data.qvel[6:].copy()
                    tau = kp * (self.q[7:] - cur_q) - kd * cur_qd
                    data.ctrl[:] = np.clip(tau, -4., 4.)
                    mujoco.mj_step(model, data)
                    viewer.sync()
    
    def deploy_remote(self, host, is_sender):
        from remote import NumpySocket
        cfg = load_config('sundaya1_real_config_half_2dof.yaml')
        heading_update = 60
        heading_counter = 0
        if not is_sender:
            remote = NumpySocket(host="0.0.0.0", port=9000, is_sender=False)
            motor_manager = MotorControllerManager(
                cfg.real_config.n_motors,
                cfg.real_config.motor_controllers,
                cfg.real_config.calibration_file,
                mode=0
            )
            motor_manager.set_positions(cfg.default_qpos, 0, 50)
            while True:
                q_recv = remote.recv()
                motor_manager.set_positions(q_recv, 0, 50)
        else:
            import pygame
            
            pygame.init()
            pygame.joystick.init()
            use_joystick = pygame.joystick.get_count() > 0
            print('Using joystick:', use_joystick)
            
            if use_joystick:
                joystick = pygame.joystick.Joystick(0)
                joystick.init()
            else:
                state = {"direction": "straight"}
                self.deploy_server(state)
            remote = NumpySocket(host=host, port=9000, is_sender=True)
            remote.send(cfg.default_qpos)
            input('start>')
            self.fsm.set_cmd(WalkCommand.STRAIGHT)
            dt = 0.03
            rate_limiter = RateLimiter(frequency=1 / dt, warn=True)
            self.fsm.start_walking = True
            while True:
                self.fsm.on_tick()
                
                if use_joystick:
                    for event in pygame.event.get():
                        if event.type == pygame.JOYAXISMOTION:
                            x_axis = joystick.get_axis(0)
                            y_axis = joystick.get_axis(1)
                            if abs(x_axis) < 0.2:
                                x_axis = 0
                            if abs(y_axis) < 0.2:
                                y_axis = 0
                            if y_axis < 0:
                                self.fsm.set_cmd(WalkCommand.STRAIGHT)
                            elif y_axis > 0:
                                self.fsm.set_cmd(WalkCommand.STOP)
                            if x_axis < 0:
                                self.fsm.set_cmd(WalkCommand.LEFT)
                            elif x_axis > 0:
                                self.fsm.set_cmd(WalkCommand.RIGHT)
                else:
                    if state["direction"] == "straight":
                        self.fsm.set_cmd(WalkCommand.STRAIGHT)
                    elif state["direction"] == "left":
                        self.fsm.set_cmd(WalkCommand.LEFT)
                    elif state["direction"] == "right":
                        self.fsm.set_cmd(WalkCommand.RIGHT)
                    elif state["direction"] == "stop":
                        self.fsm.set_cmd(WalkCommand.STOP)

                print(self.fsm.cmd)

                ik_target = self._get_targets()
                
                self.q = self.ik(ik_target, update_heading=(heading_counter % heading_update == 0))
                remote.send(self.q[7:])
                rate_limiter.sleep()
            
    def deploy_server(self, state):
        from flask import Flask, jsonify
        import threading
        app = Flask(__name__)

        GLOBAL_STATE = state

        @app.route("/")
        def index():
            return """
            <html>
            <body>
                <select onchange="update(this.value)">
                    <option value="straight">straight</option>
                    <option value="left">left</option>
                    <option value="right">right</option>
                    <option value="stop">stop</option>
                </select>

                <script>
                    function update(val) {
                        fetch("/set/" + val);
                    }
                </script>
            </body>
            </html>
            """

        @app.route("/set/<direction>")
        def set_direction(direction):
            GLOBAL_STATE["direction"] = direction
            return jsonify(GLOBAL_STATE)

        @app.route("/state")
        def state():
            return jsonify(GLOBAL_STATE)
        def run_server():
            # IMPORTANT: disable reloader or it will spawn another process/thread
            app.run(host="0.0.0.0", port=8090, debug=False, use_reloader=False)
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
         
    def deploy_controller(self):
        
        cfg = load_config('sundaya1_real_config_half_2dof.yaml')
        motor_manager = MotorControllerManager(
            cfg.real_config.n_motors,
            cfg.real_config.motor_controllers,
            cfg.real_config.calibration_file,
            mode=0
        )
        motor_manager.set_positions(cfg.default_qpos, 0, 30)
        
        input('start>')
        try:
            dt = 0.03
            rate_limiter = RateLimiter(frequency=1 / dt, warn=True)
            self.fsm.start_walking = True
            cmd = np.zeros(3)
            while True:
                self.fsm.set_cmd(self.get_joystick_cmd())
                print(self.fsm.cmd)
                self.fsm.on_tick()
                
                com_pos_nominal = self.fsm.stance.com.position
                theta = self.fsm.footstep_generator.ref_theta
                fwd = np.array([np.sin(theta), np.cos(theta)])     # world XY forward
                lat = np.array([np.cos(theta), -np.sin(theta)])    # world XY lateral/right
                center_xy = 0.5 * (
                    self.fsm.stance.left_foot.position[:2]
                    + self.fsm.stance.right_foot.position[:2]
                )
                d_xy = com_pos_nominal[:2] - center_xy
                d_f = d_xy @ fwd
                d_l = d_xy @ lat
                lateral_scale = 0.6
                com_xy = center_xy + d_f * fwd + (lateral_scale * d_l) * lat
                com_pos = np.array([com_xy[0], com_xy[1], com_pos_nominal[2]])

                start_time = time.time()
                self.q, success = self.ik(IKTarget(
                    left_foot_pose=self.fsm.stance.left_foot,
                    right_foot_pose=self.fsm.stance.right_foot,
                    com_pos=com_pos,
                    heading=self.fsm.footstep_generator.ref_theta
                ))
                # print(f"IK solve time: {time.time() - start_time:.3f}s")
                
                # rate_limiter_inner = RateLimiter(frequency=1 / 0.002, warn=True)
                # for _ in range(int(dt / 0.002)):
                #     duty = 32 * (self.q[7:] - motor_manager.get_state()[0]) - 1.5 * motor_manager.get_state()[1]
                #     motor_manager.set_duty(duty * 200)
                #     rate_limiter_inner.sleep()
                
                motor_manager.set_positions(self.q[7:], 0, 50)
                # rate_limiter.sleep()
        except KeyboardInterrupt:
            motor_manager.disable_torque()
        
           
    def deploy(self):
        from flask import Flask, jsonify
        import threading
        app = Flask(__name__)

        GLOBAL_STATE = {"direction": "straight"}

        @app.route("/")
        def index():
            return """
            <html>
            <body>
                <select onchange="update(this.value)">
                    <option value="straight">straight</option>
                    <option value="left">left</option>
                    <option value="right">right</option>
                </select>

                <script>
                    function update(val) {
                        fetch("/set/" + val);
                    }
                </script>
            </body>
            </html>
            """

        @app.route("/set/<direction>")
        def set_direction(direction):
            GLOBAL_STATE["direction"] = direction
            return jsonify(GLOBAL_STATE)

        @app.route("/state")
        def state():
            return jsonify(GLOBAL_STATE)
        def run_server():
            # IMPORTANT: disable reloader or it will spawn another process/thread
            app.run(host="0.0.0.0", port=8000, debug=False, use_reloader=False)
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        cfg = load_config('sundaya1_real_config_half_2dof.yaml')
        motor_manager = MotorControllerManager(
            cfg.real_config.n_motors,
            cfg.real_config.motor_controllers,
            cfg.real_config.calibration_file,
            mode=0
        )
        motor_manager.set_positions(cfg.default_qpos, 0, 30)
        
        input('start>')
        try:
            dt = 0.03
            rate_limiter = RateLimiter(frequency=1 / dt, warn=True)
            self.fsm.start_walking = True
            while True:
                if GLOBAL_STATE["direction"] == "straight":
                    self.fsm.set_cmd(WalkCommand.STRAIGHT)
                elif GLOBAL_STATE["direction"] == "left":
                    self.fsm.set_cmd(WalkCommand.LEFT)
                elif GLOBAL_STATE["direction"] == "right":
                    self.fsm.set_cmd(WalkCommand.RIGHT)
                self.fsm.on_tick()
                
                com_pos_nominal = self.fsm.stance.com.position
                theta = self.fsm.footstep_generator.ref_theta
                fwd = np.array([np.sin(theta), np.cos(theta)])     # world XY forward
                lat = np.array([np.cos(theta), -np.sin(theta)])    # world XY lateral/right
                center_xy = 0.5 * (
                    self.fsm.stance.left_foot.position[:2]
                    + self.fsm.stance.right_foot.position[:2]
                )
                d_xy = com_pos_nominal[:2] - center_xy
                d_f = d_xy @ fwd
                d_l = d_xy @ lat
                lateral_scale = 0.6
                com_xy = center_xy + d_f * fwd + (lateral_scale * d_l) * lat
                com_pos = np.array([com_xy[0], com_xy[1], com_pos_nominal[2]])

                start_time = time.time()
                self.q = self.ik(IKTarget(
                    left_foot_pose=self.fsm.stance.left_foot,
                    right_foot_pose=self.fsm.stance.right_foot,
                    com_pos=com_pos,
                    heading=self.fsm.footstep_generator.ref_theta
                ))
                # print(f"IK solve time: {time.time() - start_time:.3f}s")
                
                # rate_limiter_inner = RateLimiter(frequency=1 / 0.002, warn=True)
                # for _ in range(int(dt / 0.002)):
                #     duty = 32 * (self.q[7:] - motor_manager.get_state()[0]) - 1.5 * motor_manager.get_state()[1]
                #     motor_manager.set_duty(duty * 200)
                #     rate_limiter_inner.sleep()
                
                motor_manager.set_positions(self.q[7:], 0, 50)
                # rate_limiter.sleep()
        except KeyboardInterrupt:
            motor_manager.disable_torque()
        
        
        
    
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='view', choices=['view', 'simulate', 'deploy', 'remote', 'controller'], help='Mode: view, simulate, deploy, remote, controller')
    parser.add_argument('--sender', action='store_true', help='Deploy via remote sender')
    parser.add_argument('--host', type=str, default='', help='Remote host')
    args = parser.parse_args()
    
    robot = Robot()
    if args.mode == 'view':
        robot.visualize()
    elif args.mode == 'simulate':
        robot.simulate()
    elif args.mode == 'deploy':
        robot.deploy()
    elif args.mode == "remote":
        robot.deploy_remote(args.host, args.sender)
    elif args.mode == "controller":
        robot.deploy_controller()
