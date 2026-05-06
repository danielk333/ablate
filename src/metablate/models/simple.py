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
        return PosVelResults(
            position=parameters[0:3, None] + parameters[3:6, None] * times[None, :],
            velocity=np.repeat(parameters[3:6], len(times)).reshape(3, len(times)),
        )


class LinearConstantAcceleration(MeteorModel[LCA_InitState, Options, PosVelResults]):
    def __init__(self, options: Options = Options()):
        self.options = options

    def run(self, times: NDArray_N, parameters: LCA_InitState) -> PosVelResults:
        v0 = np.linalg.norm(parameters[3:6])
        v_hat = parameters[3:6] / v0
        disp = v0 * times + 0.5 * parameters[6] * times**2
        return PosVelResults(
            position=parameters[0:3, None] + v_hat[:, None] * disp[None, :],
            velocity=parameters[3:6, None] + parameters[6] * times[None, :],
        )
