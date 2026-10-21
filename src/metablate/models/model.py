from abc import ABC, abstractmethod
from typing import Generic

from ..types import Op, P, R


class MeteorModel(ABC, Generic[P, Op, R]):

    def __init__(self, options: Op) -> None:
        self.options = options

    @abstractmethod
    def run(self, parameters: P) -> R:
        pass


