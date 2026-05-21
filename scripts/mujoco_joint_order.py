import argparse
from viberobotics.configs.config import load_config
import mujoco

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="sundaya1_real_config_arm_only.yaml", help='Path to config file')
    args = parser.parse_args()

    cfg = load_config(args.config)
    mujoco_path = cfg.sim_config.asset_path
    
    model = mujoco.MjModel.from_xml_path(mujoco_path)
    print('Mujoco joint order:')
    for i in range(1, model.njnt):
        joint_name = model.joint(i).name
        print(f"Joint {i - 1}: {joint_name}")