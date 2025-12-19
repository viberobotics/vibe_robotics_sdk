from enum import Enum
from pathlib import Path

class ControlMode(Enum):
    NONE = 0
    PD_STAND = 1
    RL = 2

ROOT_DIR = Path(__file__).parent
ASSET_DIR = ROOT_DIR / "assets"
CONFIG_DIR = ROOT_DIR / "configs"