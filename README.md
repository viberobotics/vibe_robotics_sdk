# viberobotics-python

Python package for Vibe Robotics.

## Installation

### Option 1: conda environment file (recommended)

```bash
conda env create -f environment.yml
conda activate viberobotics
```

### Option 2: manual setup

```bash
conda create -n vibe python=3.10
conda activate vibe
conda install -c conda-forge pinocchio=3.9.0
pip install -r requirements.txt
pip install -e .
```

## Requirements

- Python 3.10
- See `requirements.txt` and `environment.yml` for full dependency lists
