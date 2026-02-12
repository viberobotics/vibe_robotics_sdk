import mujoco
import viser
from viser.extras import ViserUrdf
from pathlib import Path
from viberobotics.constants import ASSET_DIR
import pinocchio as pin
import numpy as np
import pink
from scipy.spatial.transform import Rotation as R
from pink import solve_ik
from pink.tasks import ComTask, FrameTask, PostureTask
import qpsolvers
from loop_rate_limiters import RateLimiter

MUJOCO_TO_URDF = [
    0,   # head_yaw
    6,   # right_shoulder_pitch
    5,   # right_shoulder_roll
    4,   # right_shoulder_yaw
    3,   # right_elbow_pitch
    2,   # right_wrist_yaw
    1,   # right_hand
    12,  # right_hip_roll
    11,  # right_hip_pitch
    10,  # right_hip_yaw
    9,   # right_knee
    8,   # right_ankle_pitch
    7,   # right_ankle_roll
    17,  # left_shoulder_pitch
    16,  # left_shoulder_roll
    15,  # left_elbow_pitch
    14,  # left_wrist_yaw
    13,  # left_hand
    23,  # left_hip_roll
    22,  # left_hip_pitch
    21,  # left_hip_yaw
    20,  # left_knee
    19,  # left_ankle_pitch
    18,  # left_ankle_roll
]

URDF_TO_MUJOCO = [
    0,   # head_yaw
    6,   # right_hand
    5,   # right_wrist_yaw
    4,   # right_elbow_pitch
    3,   # right_shoulder_yaw
    2,   # right_shoulder_roll
    1,   # right_shoulder_pitch
    12,  # right_ankle_roll
    11,  # right_ankle_pitch
    10,  # right_knee
    9,   # right_hip_yaw
    8,   # right_hip_pitch
    7,   # right_hip_roll
    17,  # left_hand
    16,  # left_wrist_yaw
    15,  # left_elbow_pitch
    14,  # left_shoulder_roll
    13,  # left_shoulder_pitch
    23,  # left_ankle_roll
    22,  # left_ankle_pitch
    21,  # left_knee
    20,  # left_hip_yaw
    19,  # left_hip_pitch
    18,  # left_hip_roll
]

# EE -> Hand
# p_hand = T_HAND_EE @ p_ee
T_HAND_EE = np.array([
    [ 0.02334125,  0.99972756,  0.0,  0.03040415],
    [-0.99972756,  0.02334125,  0.0, -0.01697652],
    [ 0.0,         0.0,         1.0, -0.01553490],
    [ 0.0,         0.0,         0.0,  1.0       ],
])

# Hand -> EE
# p_ee = T_EE_HAND @ p_hand
T_EE_HAND = np.array([
    [ 0.02334125, -0.99972756,  0.0, -0.01768157],
    [ 0.99972756,  0.02334125,  0.0, -0.02999961],
    [ 0.0,         0.0,         1.0,  0.01553490],
    [ 0.0,         0.0,         0.0,  1.0       ],
])

def _pad1(pos):
    return np.array([pos[0], pos[1], pos[2], 1.0])

if __name__ == '__main__':
    xml_path = ASSET_DIR / 'mujoco/SundayA1_new_arm/robot.xml'
    urdf_path = ASSET_DIR / 'urdf/SundayA1_new_arm/robot.urdf'
    
    default_qpos = np.zeros(24,)
    default_qpos[1] = 0.7
    
    server = viser.ViserServer()
    viser_urdf = ViserUrdf(
        server,
        urdf_or_path=urdf_path,
        load_meshes=True,
        load_collision_meshes=True,
        root_node_name="/sunday_a1"
    )
    viser_urdf.update_cfg(default_qpos[MUJOCO_TO_URDF])
    
    # model = mujoco.MjModel.from_xml_path(xml_path.as_posix())
    # print('mujoco joint order:')
    # for i in range(model.njnt):
    #     if i == 0:
    #         continue
    #     joint_name = model.joint(i).name
    #     print(f"Joint {i - 1}: {joint_name}")
    # print('urdf joint order:')
    # for i, joint_name in enumerate(viser_urdf.get_actuated_joint_names()):
    #     print(f"Joint {i}: {joint_name}")
    
    robot = pin.RobotWrapper.BuildFromMJCF(
        filename=xml_path.as_posix(),
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
    t_w_hand = np.eye(4)
    t_w_hand[:3, 3] = right_hand_pos
    t_w_hand[:3, :3] = right_hand_rot.as_matrix()
    t_w_ee = t_w_hand @ T_HAND_EE
    right_hand_control = server.scene.add_transform_controls(
        f"right_hand_control",
        scale=0.1,
        position=t_w_ee[:3, 3],
        wxyz=R.from_matrix(t_w_ee[:3, :3]).as_quat(scalar_first=True),
    )
    @right_hand_control.on_update
    def _(_):
        t_w_ee = np.eye(4)
        t_w_ee[:3, 3] = right_hand_control.position
        t_w_ee[:3, :3] = R.from_quat(right_hand_control.wxyz, scalar_first=True).as_matrix()
        t_w_hand = t_w_ee @ T_EE_HAND
        tasks[0].transform_target_to_world.translation = t_w_hand[:3, 3]
        tasks[0].transform_target_to_world.rotation = t_w_hand[:3, :3]
        
    
    dt = 1 / 200
    rate = RateLimiter(frequency=200.0, warn=False)
    dt = rate.period
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
        q[6] = 0.
        viser_urdf.update_cfg(q[MUJOCO_TO_URDF])
        rate.sleep()