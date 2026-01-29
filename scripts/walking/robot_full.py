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
            xml_path=get_asset_path('mujoco/SundayA1_full_2dof_b_new/robot.xml'),
            left_foot_name='left_foot0120',
            right_foot_name='right_foot0120',
            foot_size=np.array([0.064, 0.09, 0.005]) * 2
        )
        self.walk_config = WalkConfig(
            ssp_duration=0.7,
            dsp_duration=0.07,
            step_length=0.02,
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
       max_iters: int = 600,
       tol: float = 1e-3,
       damping: float = 3e-2,
       step: float = 0.01,
       update_heading: bool=False) -> np.ndarray:
        locked = np.array([1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 0], dtype=int)  + 7         # e.g. [10, 11, 12]
        free = np.array([i for i in range(self.model.nv) if i not in set(locked)], dtype=int)
        # print('free', free)
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
        q[4 + 7] = -0.6
        q[15 + 7] = -0.6
        

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
            Jf = J[:, free]

            A = (Jf @ Jf.T) + (damping ** 2) * np.eye(Jf.shape[0])
            dq_free = Jf.T @ np.linalg.solve(A, e)
            dq = np.zeros(model.nv)
            dq[free] = dq_free

            q = pin.integrate(model, q, step * dq)

        return q + np.concatenate([offset, np.zeros(self.nq - 3)])
    
    
    
    def visualize(self):
        visualizer = ViserVisualizer(get_asset_path('urdf/SundayA1_full_2dof_b/robot.urdf'))
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
        
        cmd_dropdown = visualizer.server.gui.add_dropdown('command', ['straight', 'left', 'right', 'stop'], initial_value='straight')
        @cmd_dropdown.on_update
        def _(_) -> None:
            cmd_str = cmd_dropdown.value
            if cmd_str == 'straight':
                self.fsm.set_cmd(WalkCommand.STRAIGHT)
            elif cmd_str == 'left':
                self.fsm.set_cmd(WalkCommand.LEFT)
            elif cmd_str == 'right':
                self.fsm.set_cmd(WalkCommand.RIGHT)
            elif cmd_str == 'stop':
                self.fsm.set_cmd(WalkCommand.STOP)
        
        t_slider = visualizer.server.gui.add_number('t', 0., min=0., max=100., step=0.01, disabled=True)
        
        left_foot_marker = visualizer.server.scene.add_icosphere('/left_foot_marker', radius=0.02, color=(255, 0, 0))
        right_foot_marker = visualizer.server.scene.add_icosphere('/right_foot_marker', radius=0.02, color=(0, 255, 0))
        com_marker = visualizer.server.scene.add_icosphere('/com_marker', radius=0.02, color=(0, 0, 255))
        
        dt = 0.03
        rate_limiter = RateLimiter(frequency=1/dt, warn=False)
        line_segments = []
        heading_update = 60
        heading_counter = 0
        while True:
            self.fsm.on_tick()
            t_slider.value += dt
            
            left_foot_marker.position = self.fsm.stance.left_foot.position
            right_foot_marker.position = self.fsm.stance.right_foot.position
            com_marker.position = self.fsm.stance.com.position

            
            self.q = self.ik(IKTarget(
                left_foot_pose=self.fsm.stance.left_foot,
                right_foot_pose=self.fsm.stance.right_foot,
                com_pos=self.fsm.stance.com.position,
                heading=self.fsm.footstep_generator.ref_theta
            ), update_heading=(heading_counter % heading_update == 0))
            heading_counter += 1
            
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
        from scripts.walking.remote import NumpySocket
        if not is_sender:
            remote = NumpySocket(host="0.0.0.0", port=9000, is_sender=True)
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
            remote = NumpySocket(host=host, port=9000, is_sender=True)
            cfg = load_config('sundaya1_real_config.yaml')
            remote.send(cfg.default_qpos)
            input('start>')
            self.fsm.set_cmd(WalkCommand.STRAIGHT)
            dt = 0.03
            rate_limiter = RateLimiter(frequency=1 / dt, warn=True)
            self.fsm.start_walking = True
            while True:
                self.fsm.on_tick()

                self.q = self.ik(IKTarget(
                    left_foot_pose=self.fsm.stance.left_foot,
                    right_foot_pose=self.fsm.stance.right_foot,
                    com_pos=self.fsm.stance.com.position,
                    heading=self.fsm.footstep_generator.ref_theta
                ))
                remote.send(self.q[7:])
                rate_limiter.sleep()
    
    def deploy(self):
        cfg = load_config('sundaya1_real_config.yaml')
        motor_manager = MotorControllerManager(
            cfg.real_config.n_motors,
            cfg.real_config.motor_controllers,
            cfg.real_config.calibration_file,
            mode=0
        )
        motor_manager.set_positions(cfg.default_qpos, 0, 50)
        # motor_manager.set_kp_kd(32, 32)
        input('start>')
        self.fsm.set_cmd(WalkCommand.STRAIGHT)
        try:
            dt = 0.03
            rate_limiter = RateLimiter(frequency=1 / dt, warn=True)
            self.fsm.start_walking = True
            while True:
                self.fsm.on_tick()

                self.q = self.ik(IKTarget(
                    left_foot_pose=self.fsm.stance.left_foot,
                    right_foot_pose=self.fsm.stance.right_foot,
                    com_pos=self.fsm.stance.com.position,
                    heading=self.fsm.footstep_generator.ref_theta
                ))
                
                # rate_limiter_inner = RateLimiter(frequency=1 / 0.002, warn=True)
                # for _ in range(int(dt / 0.002)):
                #     duty = 32 * (self.q[7:] - motor_manager.get_state()[0]) - 1.5 * motor_manager.get_state()[1]
                #     motor_manager.set_duty(duty * 200)
                #     rate_limiter_inner.sleep()
                
                motor_manager.set_positions(self.q[7:], 0, 50)
                rate_limiter.sleep()
        except KeyboardInterrupt:
            # motor_manager.set_positions(cfg.default_qpos, 0, 50)
            print('done')
        return
        time.sleep(1.)
        poses = [
            np.array([-0.00306797, -0.55836892,  0.00613594, -0.00613594, -0.90504861,
       -0.00460196, -0.078233  ,  0.02300978,  0.00153399,  0.0107379 ,
       -0.06289315,  0.03528166, -0.00306797,  0.00153399, -0.00153399,
       -0.00306797,  0.00613594,  0.04908729,  0.02454376, -0.00153399,
        0.05215526,  0.02914572,  0.05675721]),
            np.array([-0.00306797, -0.58598065,  0.        , -0.02914572, -0.89737868,
       -1.07992244, -0.07209706,  0.02147579,  0.00460196,  0.0107379 ,
       -0.06902909, -0.01380587, -0.00153399,  0.00153399, -0.00153399,
       -0.00306797,  0.00613594, -0.01227188, -0.00766993,  0.00306797,
       -0.00460196,  0.03221369,  0.0306797 ]),
            np.array([-0.00153399, -0.58598065,  0.00460196, -0.03221369, -0.89737868,
        0.00920391, -0.06289315,  0.02147579,  0.00460196,  0.01533985,
       -0.06902909, -0.0107379 , -0.00153399,  0.00153399, -0.00153399,
       -0.00306797,  0.00613594, -0.0107379 , -0.0107379 ,  0.00153399,
        0.0475533 , -0.02147579,  0.02607775]),
            np.array([-1.53398514e-03, -1.76101005e+00,  4.60195541e-03, -3.37476730e-02,
       -9.05048609e-01,  7.66992569e-03, -5.52232265e-02,  2.30097771e-02,
        4.60195541e-03,  1.99418068e-02, -6.90290928e-02, -1.22718811e-02,
       -0.00000000e+00,  1.53398514e-03, -1.53398514e-03, -1.53398514e-03,
        6.13594055e-03, -1.84078217e-02, -1.07378960e-02, -0.00000000e+00,
        5.67572117e-02, -2.45437622e-02,  2.60777473e-02]),
            np.array([-1.53398514e-03, -1.76254392e+00,  9.15786505e-01, -3.37476730e-02,
       -9.03514624e-01,  7.66992569e-03, -5.98251820e-02,  2.30097771e-02,
        4.60195541e-03,  4.14175987e-02, -6.90290928e-02, -7.66992569e-03,
        1.53398514e-03,  1.53398514e-03, -1.53398514e-03, -1.53398514e-03,
        6.13594055e-03, -4.44853306e-02, -4.60195541e-03, -0.00000000e+00,
        5.67572117e-02, -3.37476730e-02,  3.22136879e-02]),
            np.array([-1.53398514e-03, -1.76101005e+00, -4.60193157e-02, -3.52816582e-02,
       -9.03514624e-01,  7.66992569e-03, -6.28931522e-02,  2.30097771e-02,
        4.60195541e-03,  4.14175987e-02, -6.90290928e-02, -6.13594055e-03,
        1.53398514e-03,  1.53398514e-03, -1.53398514e-03, -1.53398514e-03,
        6.13594055e-03, -4.29513454e-02, -4.60195541e-03,  1.53398514e-03,
        5.67572117e-02, -3.52816582e-02,  3.37476730e-02])
        ]
        for pose in poses:
            motor_manager.set_positions(pose, 0, 50)
            input('>')
        
        
        
    
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='view', choices=['view', 'simulate', 'deploy'], help='Mode: view, simulate, deploy')
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