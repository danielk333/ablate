import time
from dataclasses import dataclass
import numpy as np
from .model import MeteorModel
from ..types import Options, NDArray_M, Results, NDArray_3xN, NDArray_N


LCV_InitState = NDArray_M
"""(m,) shaped ndarray with index:
    0-2: initial ECEF position
    3-5: initial ECEF velocity vector
"""


@dataclass
class PosVelResults(Results):
    position: NDArray_3xN
    velocity: NDArray_3xN


LCA_InitState = NDArray_M
"""(m,) shaped ndarray with index:
    0-2: initial ECEF position
    3-5: initial ECEF velocity vector
    6: initial acceleration
"""


class LinearConstantVelocity(MeteorModel[LCV_InitState, Options, PosVelResults]):
    def __init__(self, options: Options = Options()):
        self.options = options

    def run(self, times: NDArray_N, parameters: LCV_InitState) -> PosVelResults:
        t0 = time.perf_counter()
        position = parameters[0:3, None] + parameters[3:6, None] * times[None, :]
        velocity = np.repeat(parameters[3:6], len(times)).reshape(3, len(times))
        return PosVelResults(
            runtime=time.perf_counter() - t0,
            position=position,
            velocity=velocity,
        )


class LinearConstantAcceleration(MeteorModel[LCA_InitState, Options, PosVelResults]):
    def __init__(self, options: Options = Options()):
        self.options = options

    def run(self, times: NDArray_N, parameters: LCA_InitState) -> PosVelResults:
        t0 = time.perf_counter()
        v0 = np.linalg.norm(parameters[3:6])
        v_hat = parameters[3:6] / v0
        disp = v0 * times + 0.5 * parameters[6] * times**2
        position = parameters[0:3, None] + v_hat[:, None] * disp[None, :]
        velocity = parameters[3:6, None] + parameters[6] * times[None, :]
        return PosVelResults(
            runtime=time.perf_counter() - t0,
            position=position,
            velocity=velocity,
        )
