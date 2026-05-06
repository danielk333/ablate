from abc import ABC, abstractmethod
from typing import Generic
from ..types import NDArray_N, P, Op, R

class MeteorModel(ABC, Generic[P, Op, R]):

    def __init__(self, options: Op):
        self.options = options

    @abstractmethod
    def run(self, times: NDArray_N, parameters: P) -> R:
        pass


