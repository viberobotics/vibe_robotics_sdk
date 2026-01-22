from viberobotics.constants import ASSET_DIR, CONFIG_DIR

from dataclasses import dataclass
import numpy as np
from typing import Union, List
import yaml

@dataclass
class SundayA1SimConfig:
    dt: float = 0.002
    asset_path: str = ""
    urdf_path: str = ""

@dataclass
class SerialConfig:
    port: str = '/dev/ttyACM0'
    baudrate: int = 1000000

@dataclass
class MotorControllerConfig:
    name: str = "DefaultMotorController"
    motor_ids: List[int] = None
    serial_config: SerialConfig = None
    sign_change: List[int] = None
    sim_idxs: List[int] = None

@dataclass
class SundayA1RealConfig:
    imu_port: SerialConfig
    motor_controllers: List[MotorControllerConfig] = None
    motor_order: dict[int, int] = None # motor_orders[real id] = sim index
    
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
    num_observations: int = 0
    horizon: int = 10
    policy_interval: int = 0.02 # 10 * dt
    action_scale: float = 0.25

@dataclass
class SundayA1Config:
    default_qpos: np.ndarray
    sim_config: SundayA1SimConfig
    real_config: SundayA1RealConfig
    control_config: SundayA1ControlConfig
    policy_config: SundayA1PolicyConfig
    

def load_config(config_path, from_config_dir=True):
    config_path = CONFIG_DIR / config_path if from_config_dir else config_path
    with open(config_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    sim_cfg = SundayA1SimConfig(**cfg_dict.get('sim', {}))
    real_cfg = SundayA1RealConfig(
        imu_port=SerialConfig(**cfg_dict.get('real', {}).get('imu_port', {})),
        motor_controllers=[
            MotorControllerConfig(**{**motor, 'serial_config': SerialConfig(**motor.get('serial_config', {}))}) for motor in cfg_dict.get('real', {}).get('motor_controllers', [])
        ],
        motor_order=cfg_dict.get('real', {}).get('motor_order', {})
    )
    control_cfg = SundayA1ControlConfig(**cfg_dict.get('control', {}))
    policy_cfg = SundayA1PolicyConfig(**cfg_dict.get('policy', {}))
    policy_cfg.num_observations = policy_cfg.num_actions * 3 + 9
    default_qpos = np.array(cfg_dict.get('default_qpos', np.zeros(21)), dtype=np.float32)
    config = SundayA1Config(
        default_qpos=default_qpos,
        sim_config=sim_cfg,
        real_config=real_cfg,
        control_config=control_cfg,
        policy_config=policy_cfg
    )
    config.sim_config.asset_path = (ASSET_DIR / cfg_dict['sim']['asset_path']).as_posix()
    config.policy_config.model_path = (ASSET_DIR / cfg_dict['policy']['model_path']).as_posix()
    return config
    
if __name__ == "__main__":
    print(CONFIG_DIR / 'sundaya1_real_config.yaml')
    config = load_config(CONFIG_DIR / 'sundaya1_real_config.yaml')
    print(config)