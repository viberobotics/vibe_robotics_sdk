from viberobotics.sensor.d435i import RealsenseD435i
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
import qpsolvers
import pink
from pink import solve_ik
from pink.tasks import ComTask, FrameTask, PostureTask
from pink.visualization import start_meshcat_visualizer
import mujoco

def urdf_to_mj(q):
    return np.array(q)[[0, 
                       5, 4, 3, 2, 1,
                       11, 10, 9, 8, 7, 6,
                       16, 15, 14, 13, 12,
                       22, 21, 20, 19, 18, 17,]]
def mj_to_urdf(q):
    return np.array(q)[[0,
                       5, 4, 3, 2, 1,
                       11, 10, 9, 8, 7, 6,
                       16, 15, 14, 13, 12,
                       22, 21, 20, 19, 18, 17,]]

EE_OFFSET = [-0.01692857, -0.01473608, -0.04638411]

if __name__ == '__main__':
    
    cfg = load_config('sundaya1_real_config_arm_only.yaml')
    motor_manager = MotorControllerManager(23, cfg.real_config.motor_controllers, mode=0)
    print(motor_manager.motor_order)
    motor_manager.set_positions(np.zeros(23,), 0, 5)
    
    cam = RealsenseD435i()
    server = viser.ViserServer()
    viser_urdf = ViserUrdf(
        server,
        urdf_or_path=Path('/home/danielchen09/dc/vibe/viberobotics-python/viberobotics/assets/urdf/SundayA1_full_2dof/robot.urdf'),
        load_meshes=True,
        load_collision_meshes=False,
        root_node_name="/sunday_a1"
    )
    default_qpos = np.zeros(23,)
    viser_urdf.update_cfg(np.zeros(23,))
    
    print('mujoco joint order:')
    model = mujoco.MjModel.from_xml_path('/home/danielchen09/dc/vibe/viberobotics-python/viberobotics/assets/mujoco/SundayA1_full_2dof/scene.xml')
    for i in range(model.njnt):
        joint_name = model.joint(i).name
        print(f"Joint {i}: {joint_name}")
    print('urdf joint order:')
    for i, name in enumerate(viser_urdf.get_actuated_joint_names()):
        print(f"Joint {i}: {name}")
    
    robot = pin.RobotWrapper.BuildFromMJCF(
        filename="/home/danielchen09/dc/vibe/viberobotics-python/viberobotics/assets/mujoco/SundayA1_full_2dof/robot.xml",
        root_joint=None,
    )
    configuration = pink.Configuration(robot.model, robot.data, default_qpos)
    pin.forwardKinematics(robot.model, robot.data, default_qpos)
    pin.updateFramePlacements(robot.model, robot.data)
    right_hand_id = robot.model.getFrameId("thumb_0112")
    right_hand_pos = robot.data.oMf[right_hand_id].translation.copy()
    right_hand_rot = R.from_matrix(robot.data.oMf[right_hand_id].rotation)
    right_hand_quat = right_hand_rot.as_quat(scalar_first=True)
    
    tasks = [
        FrameTask(
            "thumb_0112",
            position_cost=4.0,
            orientation_cost=1.0,
        ),
    ]
    for task in tasks:
        task.set_target_from_configuration(configuration)
    solver = qpsolvers.available_solvers[0]
    if "proxqp" in qpsolvers.available_solvers:
        solver = "proxqp"
    
    right_hand_control = server.scene.add_transform_controls(
        f"right_hand_control",
        scale=0.1,
        position=right_hand_pos,
        wxyz=right_hand_quat,
    )
    @right_hand_control.on_update
    def _(_):
        target_pos = right_hand_control.position - np.array(EE_OFFSET)
        target_quat = right_hand_control.wxyz
        tasks[0].transform_target_to_world.translation = target_pos
        r = R.from_quat(target_quat, scalar_first=True)
        tasks[0].transform_target_to_world.rotation = r.as_matrix()
    @right_hand_control.on_drag_end
    def _(_):
        print("Offset: ", right_hand_control.position - right_hand_pos)
    
    gripper_slider = server.gui.add_slider(
        "gripper slider",
        min=0.0,
        max=0.5,
        step=0.001,
        initial_value=0.0
    )
    
    head_id = robot.model.getFrameId("head1209")
    head_pos = robot.data.oMf[head_id].translation.copy()
    head_rot = R.from_matrix(robot.data.oMf[head_id].rotation)
    head_quat = head_rot.as_quat(scalar_first=True)
    
    cam_local_pos = [-1.8908e-17, 0.00146458, -0.0981134]
    cam_local_quat = [0.866025, -0.5, -1.11022e-16, -6.02582e-17]
    
    cam_global_pos = head_pos + head_rot.apply(cam_local_pos)
    cam_global_rot = head_rot * R.from_quat(cam_local_quat, scalar_first=True)
    cam_global_quat = cam_global_rot.as_quat(scalar_first=True)
    
    cam_control = server.scene.add_transform_controls(
        f"cam_control",
        scale=0.1,
        position=cam_global_pos,
        wxyz=cam_global_quat,
    )
    
    # point cloud -> cam frame: z -> y, y -> z, x -> -x
    cam_internal_rot = R.from_matrix([[-1, 0, 0],
                                      [0, -1, 0],
                                      [0, 0, 1]])
    
    
    
    dt = 1 / 200
    rate = RateLimiter(frequency=200.0, warn=False)
    dt = rate.period
    print('start')
    while True:
        velocity = solve_ik(
            configuration,
            tasks,
            dt,
            solver=solver,
            damping=0.01,
            safety_break=False,
        )
        configuration.integrate_inplace(velocity, dt)
        q = np.array(configuration.q)
        q[5] = gripper_slider.value
        viser_urdf.update_cfg(mj_to_urdf(q).copy()) 
        
        motor_manager.set_positions(q, 0, 5)
        
        pc_xyz_internal, pc_color = cam.get_pointcloud()
        pc_xyz = cam_global_rot.apply(cam_internal_rot.apply(pc_xyz_internal)) + cam_global_pos
        server.scene.add_point_cloud(
            "/pointcloud",
            points=pc_xyz,
            colors=pc_color,
            point_size=0.002,
        )
        rate.sleep()