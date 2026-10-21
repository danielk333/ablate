"""
# Meteoroid Ablation Models
"""
from __future__ import annotations

import logging
import sys

from . import atmosphere, material, models, physics
from .atmosphere import Atmosphere
from .version import __version__


def setup_logging(level: int | str = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("metablate")
    logger.setLevel(level)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
