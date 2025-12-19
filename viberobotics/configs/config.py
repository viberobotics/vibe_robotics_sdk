from viberobotics.constants import ASSET_DIR, CONFIG_DIR

from dataclasses import dataclass
import numpy as np
from typing import Union
import yaml

@dataclass
class SundayA1SimConfig:
    dt: float = 0.002
    asset_path: str = ""

@dataclass
class SundayA1ControlConfig:
    kp_torque: Union[np.ndarray, float] = 0.
    kd_torque: Union[np.ndarray, float] = 0.
    kp_duty: Union[np.ndarray, float] = 0.
    ki_duty: Union[np.ndarray, float] = 0.
    voltage: float = 5.
    K_t: float = 0.765
    K_e: float = 1.227
    internal_resistance: float = 2.5

@dataclass
class SundayA1PolicyConfig:
    model_path: str
    num_actions: int = 21
    num_observations: int = 72
    horizon: int = 10
    policy_interval: int = 0.02 # 10 * dt
    action_scale: float = 0.25

@dataclass
class SundayA1Config:
    default_qpos: np.ndarray
    sim_config: SundayA1SimConfig
    control_config: SundayA1ControlConfig
    policy_config: SundayA1PolicyConfig
    

def load_config(config_path):
    with open(config_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    sim_cfg = SundayA1SimConfig(**cfg_dict.get('sim', {}))
    control_cfg = SundayA1ControlConfig(**cfg_dict.get('control', {}))
    policy_cfg = SundayA1PolicyConfig(**cfg_dict.get('policy', {}))
    default_qpos = np.array(cfg_dict.get('default_qpos', np.zeros(21)), dtype=np.float32)
    config = SundayA1Config(
        default_qpos=default_qpos,
        sim_config=sim_cfg,
        control_config=control_cfg,
        policy_config=policy_cfg
    )
    config.sim_config.asset_path = (ASSET_DIR / cfg_dict['sim']['asset_path']).as_posix()
    config.policy_config.model_path = (ASSET_DIR / cfg_dict['policy']['model_path']).as_posix()
    return config
    
if __name__ == "__main__":
    print(CONFIG_DIR / 'sundaya1_sim_config.yaml')
    config = load_config(CONFIG_DIR / 'sundaya1_sim_config.yaml')
    print(config)