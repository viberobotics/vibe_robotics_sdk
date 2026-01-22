from viberobotics.configs.config import load_config
from viberobotics.motor.motor_controller_manager import MotorControllerManager

import numpy as np
import argparse
from pathlib import Path

import viser
from viser.extras import ViserUrdf

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='viberobotics/configs/sundaya1_real_config_leg_only.yaml', help='Config file path')
    args = parser.parse_args()
    cfg = load_config(args.config)
    
    server = viser.ViserServer()
    robot_base = server.scene.add_frame("/sunday_a1", show_axes=False)
    robot_base.position = (0, 0, 0.207)
    viser_urdf = ViserUrdf(
        server,
        urdf_or_path=Path(cfg.sim_config.urdf_path),
        load_meshes=True,
        load_collision_meshes=False,
        root_node_name="/sunday_a1"
    )
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
    
    viser_urdf.get_actuated_joint_limits()
    