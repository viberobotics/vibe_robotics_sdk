import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
from viberobotics.constants import ASSET_DIR
import numpy as np
import hppfcl as fcl
from scipy.spatial.transform import Rotation as R
from pink.tasks import ComTask, FrameTask, PostureTask
import pink
from pink import solve_ik
import qpsolvers
import viser
from viser.extras import ViserUrdf
from pathlib import Path
from loop_rate_limiters import RateLimiter
import mujoco
from pyroboplan.planning.rrt import RRTPlanner, RRTPlannerOptions
import time

END_POS_DEFUALT = np.array([
    0.0, 0.1, 0.5, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0
])
FREE_JOINTS_IDX_MJ = [1, 2, 3, 4]

EE_OFFSET_LOCAL = [-0.01692857, -0.01473608, -0.04638411]
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

if __name__ == "__main__":
    pc_xyz, pc_color = np.load("pc_xyz.npy"), np.load("pc_color.npy")
    
    robot = pin.RobotWrapper.BuildFromMJCF(
        filename=(ASSET_DIR / "mujoco/SundayA1_full_2dof_arm_manip/robot.xml").as_posix(),
        root_joint=None,
    )
    print('mujoco joint order:')
    model = mujoco.MjModel.from_xml_path('/home/danielchen09/dc/vibe/viberobotics-python/viberobotics/assets/mujoco/SundayA1_full_2dof_arm_manip/scene.xml')
    for i in range(model.njnt):
        joint_name = model.joint(i).name
        print(f"Joint {i}: {joint_name}")
    robot = robot.buildReducedRobot(list_of_joints_to_lock=[
        "head_yaw",            # 0
        "right_hand",          # 5
        "right_hip_roll",      # 6
        "right_hip_pitch",     # 7
        "right_hip_yaw",       # 8
        "right_knee",          # 9
        "right_ankle_pitch",   # 10
        "right_ankle_roll",    # 11
        "left_shoulder_roll",  # 12
        "left_shoulder_pitch", # 13
        "left_elbow_pitch",          # 14
        "left_wrist_yaw",          # 15
        "left_hand",           # 16
        "left_hip_roll",       # 17
        "left_hip_pitch",      # 18
        "left_hip_yaw",        # 19
        "left_knee",           # 20
        "left_ankle_pitch",    # 21
        "left_ankle_roll",     # 22
    ])
    
    
    collision_model = robot.collision_model
    model = robot.model
    print(model.lowerPositionLimit, model.upperPositionLimit)
    visual_model = robot.visual_model
    
    
    default_qpos_mj = np.zeros(23,)
    default_qpos_mj[5] = 0.1
    default_qpos_mj[16] = 0.1
    
    
    
    # add self collisions
    collision_model.addAllCollisionPairs()
    
    data = model.createData()
    collision_data = collision_model.createData()
    
    
    # external collisions
    octree = fcl.makeOctree(pc_xyz, 0.01)
    octree_object = pin.GeometryObject("octree", 0, pin.SE3.Identity(), octree)
    octree_object.meshColor[0] = 1.0
    collision_model.addGeometryObject(octree_object)
    
    pin.computeCollisions(model, data, collision_model, collision_data, default_qpos_mj[FREE_JOINTS_IDX_MJ], False)
    for k in range(len(collision_model.collisionPairs)):
        cr = collision_data.collisionResults[k]
        cp = collision_model.collisionPairs[k]
        if cr.isCollision():
            print(
                "collision between:",
                collision_model.geometryObjects[cp.first].name,
                " and ",
                collision_model.geometryObjects[cp.second].name,
            )
            

    
    pin.forwardKinematics(model, data, END_POS_DEFUALT[FREE_JOINTS_IDX_MJ])
    pin.updateFramePlacements(model, data)
    
    
    right_hand_id = robot.model.getFrameId("thumb_0112")
    right_hand_pos = robot.data.oMf[right_hand_id].translation.copy()
    right_hand_rot = R.from_matrix(robot.data.oMf[right_hand_id].rotation)
    right_hand_quat = right_hand_rot.as_quat(scalar_first=True)
    
    configuration = pink.Configuration(robot.model, robot.data, default_qpos_mj[FREE_JOINTS_IDX_MJ])
    tasks = [
        FrameTask(
            "thumb_0112",
            position_cost=4.0,
            orientation_cost=0.05,
        ),
    ]
    for task in tasks:
        task.set_target_from_configuration(configuration)
    solver = qpsolvers.available_solvers[0]
    if "proxqp" in qpsolvers.available_solvers:
        solver = "proxqp"
    
    viz = MeshcatVisualizer(model, collision_model, visual_model, data=data)
    viz.initViewer(open=False)
    viz.loadViewerModel()
    viz.displayCollisions(True)
    viz.displayVisuals(False)
    
    rrt_planner_option = RRTPlannerOptions(
        max_step_size=0.05,
        max_connection_dist=5.0,
        rrt_connect=False,
        bidirectional_rrt=False,
        rrt_star=True,
        max_rewire_dist=np.inf,
        max_planning_time=5.0,
        fast_return=True,
        goal_biasing_probability=0.15,
        collision_distance_padding=0.,
    )
    planner = RRTPlanner(model, collision_model, options=rrt_planner_option)
    
    q_start = default_qpos_mj.copy()
    q_goal = END_POS_DEFUALT.copy()
    # q_goal = urdf_to_mj(q_goal)
    print(q_goal[FREE_JOINTS_IDX_MJ])
    q_path = planner.plan(q_start[FREE_JOINTS_IDX_MJ], q_goal[FREE_JOINTS_IDX_MJ])
    print(q_path)
    
    while True:
        # for tree in [planner.start_tree, planner.goal_tree]:
        #     for node in tree.nodes:
        #         viz.display(node.q)
        #         time.sleep(0.1)
        pass