"""The basic class structures for implementing a ablation model.

"""

from abc import abstractmethod
import logging
from configparser import ConfigParser
from typing import Union
from pathlib import Path

logger = logging.getLogger(__name__)


class AblationModel:
    """Base-class for ablation models

    The `run` method is expected to return the results from the ablation modelling
    """

    DEFAULT_CONFIG = {}

    def __init__(self, config: Union[dict, str, Path, ConfigParser]):
        self.config = ConfigParser()
        self.config.read_dict(self.DEFAULT_CONFIG)

        if isinstance(config, ConfigParser):
            for section in config.sections():
                config.read_dict({
                    section: {key: val for key, val in config.items(section=section)}
                })
        elif isinstance(config, dict):
            self.config.read_dict(config)
        elif isinstance(config, (str, Path)):
            self.config.read_file(config)
        else:
            raise TypeError(f"Unexpected config type: {type(config)}")

    @abstractmethod
    def run(self, *args, **kwargs):
        pass
