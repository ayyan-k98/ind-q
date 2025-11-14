"""
Modified EPyMARL envs/__init__.py
This shows how to modify: epymarl/src/envs/__init__.py

INSTRUCTIONS:
1. Add the coverage import (line 7)
2. Add the coverage case in env_REGISTRY function (line 34-35)
"""

from functools import partial
from .multiagentenv import MultiAgentEnv
from .starcraft2.starcraft2 import StarCraft2Env
from .smacv2 import SMACv2

# ADD THIS LINE:
from .coverage import CoverageEnv

import sys
import os

def env_REGISTRY(env_name):
    """
    Returns environment class based on name.

    Available environments:
    - "sc2": StarCraft II (original SMAC)
    - "smac_v2": SMAC v2
    - "coverage": Multi-agent coverage (NEW)
    """

    if env_name == "sc2":
        return partial(StarCraft2Env, env_name=env_name)

    elif env_name == "smac_v2":
        return partial(SMACv2, env_name=env_name)

    # ADD THESE LINES:
    elif env_name == "coverage":
        return CoverageEnv

    else:
        raise ValueError(f"Unknown environment: {env_name}")
