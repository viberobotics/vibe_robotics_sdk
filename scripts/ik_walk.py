from viberobotics.configs.config import load_config
from viberobotics.motor.motor_controller_manager import MotorControllerManager

import pinocchio as pin
import viser
from viser.extras import ViserUrdf
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from loop_rate_limiters import RateLimiter
import argparse
import time

import qpsolvers
import pink
from pink import solve_ik
from pink.tasks import ComTask, FrameTask, PostureTask
from pink.visualization import start_meshcat_visualizer

def urdf_to_mj(q):
    return (np.array(q)[
        [4, 3, 2, 1, 0,
        9, 8, 7, 6, 5,]
    ] + np.array([
        0, 0.7, 0, 0, 0,
        0, -0.7, 0, 0, 0,
    ]))

def mj_to_urdf(q):
    return (np.array(q) - np.array([
        0, 0.7, 0, 0, 0,
        0, -0.7, 0, 0, 0,
    ]))[
        [4, 3, 2, 1, 0,
        9, 8, 7, 6, 5,]
    ]

RIGHT_FOOT_OFFSET = [ 0.08673298, 0.01949158, -0.07612692]
LEFT_FOOT_OFFSET = [ -0.08673298, 0.01949158, -0.07612692]
FOOT1_NAME = "foot1016"
FOOT2_NAME = "foot1016_2"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--real', action='store_true', help='Use real robot')
    parser.add_argument('--disable_ik', action='store_true', help='Use real robot')
    parser.add_argument('--config', type=str, default='viberobotics/configs/sundaya1_real_config_leg_only.yaml', help='Config file path')
    args = parser.parse_args()
    
    server = viser.ViserServer()
    
    cfg = load_config(args.config)
    
    if args.real:
        motor_manager = MotorControllerManager(cfg.real_config.motor_controllers, mode=0)
        motor_manager.set_positions(cfg.default_qpos, 0, 5)
    
    robot = pin.RobotWrapper.BuildFromMJCF(
        filename=(Path(cfg.sim_config.asset_path).parent / "robot.xml").as_posix(),
        root_joint=None,
    )
    default_q = np.concatenate([np.array([0, 0, 0., 0, 0, 0, 1]), cfg.default_qpos])
    configuration = pink.Configuration(robot.model, robot.data, default_q)
    print(default_q)
    pin.forwardKinematics(robot.model, robot.data, default_q)
    pin.updateFramePlacements(robot.model, robot.data)
    
    fid_right = robot.model.getFrameId(FOOT1_NAME)
    right_foot_pos = robot.data.oMf[fid_right].translation.copy()
    r_right = R.from_matrix(robot.data.oMf[fid_right].rotation)
    right_foot_rot = r_right.as_quat()
    
    fid_left = robot.model.getFrameId(FOOT2_NAME)
    left_foot_pos = robot.data.oMf[fid_left].translation.copy()
    r_left = R.from_matrix(robot.data.oMf[fid_left].rotation)
    left_foot_rot = r_left.as_quat()
    
    tasks = [
        FrameTask(
            FOOT1_NAME,
            position_cost=4.0,
            orientation_cost=1.0,
        ),
        FrameTask(
            FOOT2_NAME,
            position_cost=4.0,
            orientation_cost=1.0,
        ),
        FrameTask(
            "base",
            position_cost=4.0,
            orientation_cost=1.0,
        ),
    ]
    for task in tasks:
        task.set_target_from_configuration(configuration)
        
    solver = qpsolvers.available_solvers[0]
    if "proxqp" in qpsolvers.available_solvers:
        solver = "proxqp"
    
    
    robot_base = server.scene.add_frame("/sunday_a1", show_axes=False)
    robot_base.position = (0, 0, 0.207)
    viser_urdf = ViserUrdf(
        server,
        urdf_or_path=Path(cfg.sim_config.urdf_path),
        load_meshes=True,
        load_collision_meshes=False,
        root_node_name="/sunday_a1"
    )
    viser_urdf.update_cfg(mj_to_urdf(default_q[7:]))
    
    server.scene.add_grid(
        "/grid",
        width=2,
        height=2,
        position=(
            0.0,
            0.0,
            0.0,
        ),
    )
    
    right_foot_control = server.scene.add_transform_controls(
        f"/right_foot_control",
        depth_test=False,
        scale=0.1,
        disable_axes=False,
        disable_sliders=True,
        disable_rotations=True,
        visible=True,
        position=right_foot_pos + RIGHT_FOOT_OFFSET
    )
    @right_foot_control.on_update
    def _(_) -> None:
        tasks[0].transform_target_to_world.translation = right_foot_control.position - RIGHT_FOOT_OFFSET
    @right_foot_control.on_drag_end
    def _(_) -> None:
        print(f'Right foot offset: {right_foot_control.position - right_foot_pos}')
    
    left_foot_control = server.scene.add_transform_controls(
        f"/left_foot_control",
        depth_test=False,
        scale=0.1,
        disable_axes=False,
        disable_sliders=True,
        disable_rotations=True,
        visible=True,
        position=left_foot_pos + LEFT_FOOT_OFFSET
    )
    @left_foot_control.on_update
    def _(_) -> None:
        tasks[1].transform_target_to_world.translation = left_foot_control.position - LEFT_FOOT_OFFSET
    @left_foot_control.on_drag_end
    def _(_) -> None:
        print(f'Left foot offset: {left_foot_control.position - left_foot_pos}')
    
    base_control = server.scene.add_transform_controls(
        f"/base_control",
        depth_test=False,
        scale=0.2,
        disable_axes=False,
        disable_sliders=True,
        disable_rotations=True,
        visible=True,
        position=robot_base.position.copy(),
    )
    @base_control.on_update
    def _(_) -> None:
        tasks[2].transform_target_to_world.translation = base_control.position
    @base_control.on_drag_end
    def _(_) -> None:
        print(f'Base position: {base_control.position}')
    
    gui_reset_button = server.gui.add_button("reset")
    @gui_reset_button.on_click
    def _(_) -> None:
        global configuration
        print('reset')
        configuration = pink.Configuration(robot.model, robot.data, default_q)
        left_foot_control.position = left_foot_pos + LEFT_FOOT_OFFSET
        right_foot_control.position = right_foot_pos + RIGHT_FOOT_OFFSET
        base_control.position = (0, 0, 0.207)
        tasks[0].transform_target_to_world.translation = right_foot_pos
        tasks[1].transform_target_to_world.translation = left_foot_pos
        tasks[2].transform_target_to_world.translation = base_control.position
    
    left_target = server.scene.add_icosphere("/left_target", radius=0.02, color=(1, 0, 0))
    left_target.position = left_foot_pos
    
    right_target = server.scene.add_icosphere("/right_target", radius=0.02, color=(0, 1, 0))
    right_target.position = right_foot_pos
    
    body_target = server.scene.add_icosphere("/body_target", radius=0.02, color=(0, 0, 1))
    body_target.position = robot_base.position.copy()
    
    dt = 1 / 200
    rate = RateLimiter(frequency=200.0, warn=False)
    dt = rate.period
    start_time = time.time()
    
    stride = 0.01
    height = 0.03
    freq = 6.5  # Hz
    
    while True:
        t = time.time() - start_time
        y = stride * np.cos(freq * t)
        z = height * np.sin(-freq * t)
        left_target.position = left_foot_pos + LEFT_FOOT_OFFSET + np.array([0.01, y, max(z, 0)])
        
        y = stride * np.cos(freq * t + np.pi)
        z = height * np.sin(-freq * t + np.pi)
        right_target.position = right_foot_pos + RIGHT_FOOT_OFFSET + np.array([-0., y, max(z, 0)])
        
        x = 0.05 * np.sin(freq * t + np.pi / 2)
        y = 0.05 * np.cos(freq * t)
        body_target.position = np.array([x, 0.02, 0.207])
        
        
        tasks[0].transform_target_to_world.translation = right_target.position - RIGHT_FOOT_OFFSET
        tasks[1].transform_target_to_world.translation = left_target.position - LEFT_FOOT_OFFSET
        tasks[2].transform_target_to_world.translation = body_target.position
        
        velocity = solve_ik(
            configuration,
            tasks,
            dt,
            solver=solver,
            damping=0.01,
            safety_break=False,
        )
        configuration.integrate_inplace(velocity, dt)
        q = configuration.q
        if args.real:
            motor_manager.set_positions(q[7:], 0, 5)
        if not args.disable_ik:
            viser_urdf.update_cfg(mj_to_urdf(q[7:]).copy())
            robot_base.position = q[:3] + np.array([0, 0, 0.207])
            robot_base.wxyz = (q[6], q[3], q[4], q[5])
        rate.sleep()