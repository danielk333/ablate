"""
# Meteoroid Ablation Models
"""

import logging
import sys

from . import physics
from . import models
from . import atmosphere
from . import material

from .atmosphere import Atmosphere

from .version import __version__


def setup_logging(level=logging.INFO):
    logger = logging.getLogger("metablate")
    logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
