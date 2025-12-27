from viberobotics.configs.config import SundayA1PolicyConfig, SundayA1Config
from viberobotics.utils.buffer import Buffer
from viberobotics.utils.smoothing import RateLimitedSmoothing

import numpy as np
import torch

# Policy -> MuJoCo index
MJ_TO_POLICY = np.array([
    4, 8, 12, 16, 20,
    3, 7, 11, 15, 19,
    2, 6, 10, 14, 18,
    1, 5, 9, 13, 17,
    0
])

POLICY_TO_MJ = np.array([
    20,
    15, 10, 5, 0, 16,
    11, 6, 1, 17, 12,
    7, 2, 18, 8, 13,
    3, 19, 9, 14, 4
])

def mj_to_policy(x):
    return np.asarray(x)[..., MJ_TO_POLICY]
def policy_to_mj(x):
    return np.asarray(x)[..., POLICY_TO_MJ]

class Policy:
    def __init__(self, config: SundayA1Config):
        self.config = config
        self.policy_config = config.policy_config
        
        self.policy = torch.jit.load(self.policy_config.model_path)
        self.policy.eval()
        
        policy_cfg = config.policy_config
        self.buffer = Buffer(obs_dim=policy_cfg.num_observations, horizon=policy_cfg.horizon)
        self.smoothed_command = RateLimitedSmoothing(policy_cfg.policy_interval, default_value=np.zeros(3, dtype=np.float32))
        self.default_qpos = mj_to_policy(config.default_qpos)
        self.actions = np.zeros(policy_cfg.num_actions, dtype=np.float32)
    
    def inference(self,
                  dof_pos: np.ndarray,
                  dof_vel: np.ndarray,
                  base_ang_vel: np.ndarray,
                  projected_gravity: np.ndarray,
                  vx: float,
                  vy: float,
                  vyaw: float) -> np.ndarray:
        dof_pos = mj_to_policy(dof_pos)
        dof_vel = mj_to_policy(dof_vel)
        
        command = np.array([vx, vy, vyaw], dtype=np.float32)
        smoothed_command = self.smoothed_command.apply(command)
        
        obs = np.hstack([
            base_ang_vel,
            projected_gravity,
            smoothed_command,
            dof_pos - self.default_qpos,
            dof_vel,
            self.actions
        ])
        
        self.buffer.add(obs)
        if not self.buffer.is_full():
            return policy_to_mj(self.default_qpos)

        obs_horizon = self.buffer.get().astype(np.float32, copy=False)
        self.actions[:] = self.policy(torch.from_numpy(obs_horizon).unsqueeze(0)).detach().numpy()
        dof_targets = self.default_qpos + self.policy_config.action_scale * self.actions
        # print(np.round(obs_horizon, 3))
        # print(dof_targets, self.policy_config.action_scale, self.actions)
        # input()
        return policy_to_mj(dof_targets)