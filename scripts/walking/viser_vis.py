import viser
from viser.extras import ViserUrdf
import numpy as np
from pathlib import Path
import mujoco

class ViserVisualizer:
    def __init__(self, urdf_path: str):
        self.server = viser.ViserServer()
        self.robot_base = self.server.scene.add_frame("/robot_base", show_axes=False)
        self.robot_base.position = (0, 0, 0.2125)
        self.viser_urdf = ViserUrdf(
            self.server,
            urdf_or_path=Path(urdf_path),
            load_meshes=True,
            load_collision_meshes=False,
            root_node_name="/robot_base"
        )
        # print('mujoco joint order:')
        # model = mujoco.MjModel.from_xml_path('/home/danielchen09/dc/vibe/viberobotics-python/viberobotics/assets/mujoco/SundayA1_ankle_2dof_new/scene.xml')
        # for i in range(model.njnt):
        #     joint_name = model.joint(i).name
        #     print(f"Joint {i - 1}: {joint_name}")
        # print('urdf joint order:')
        # for i, name in enumerate(self.viser_urdf.get_actuated_joint_names()):
        #     print(f"Joint {i}: {name}")
        
        self.njnt = len(self.viser_urdf.get_actuated_joint_names())
        self.viser_urdf.update_cfg(self._urdf_to_mj(np.zeros(self.njnt)))
        self.server.scene.add_grid(
            "/grid",
            width=2,
            height=2,
            position=(
                0.0,
                0.0,
                0.0,
            ),
        )
    
    def set_state(self, q):
        self.viser_urdf.update_cfg(self._mj_to_urdf(q[7:]))
        positions = q[0:3]
        orientation = q[3:7]
        self.robot_base.position = positions + np.array([0.0, 0.0, 0.2125])
        self.robot_base.wxyz = orientation[[3, 0, 1, 2]]
    
    def add_box(self, name: str, size: np.ndarray, position: np.ndarray, color: np.ndarray):
        return self.server.scene.add_box(
            name,
            position=position,
            dimensions=size,
            color=color,
        )
        
    def _urdf_to_mj(self, q):
        if len(q) < 23:
            return np.array(q)[
                [5, 4, 3, 2, 1, 0,
                11, 10, 9, 8, 7, 6,]
            ]
        return np.array(q)[
            [0, 5, 4, 3, 2, 1,
            11, 10, 9, 8, 7, 6,
            16, 15, 14, 13, 12,
            22, 21, 20, 19, 18, 17,]
        ]
    
    def _mj_to_urdf(self, q):
        if len(q) < 23:
            return np.array(q)[
                [5, 4, 3, 2, 1, 0,
                11, 10, 9, 8, 7, 6,]
            ]
        return np.array(q)[
            [0, 5, 4, 3, 2, 1,
            11, 10, 9, 8, 7, 6,
            16, 15, 14, 13, 12,
            22, 21, 20, 19, 18, 17,]
        ]