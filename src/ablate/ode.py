#!/usr/bin/env python

"""A type of ablation models that can be solved as IVP ODE's

"""

from abc import abstractmethod
import logging

from scipy.integrate import solve_ivp

from .units import ureg
from .core import AblationModel

logger = logging.getLogger(__name__)


class ScipyODESolve(AblationModel):
    """A template class for a type of ablation model that can be solved as an IVP ODE
    using `scipy.integrate.solve_ivp`. Implements the 'integrate` method that can be
    called when constructing the `run` method. The integrate method uses the `rhs`
    method to define the right-hand-side of the differential equation system.

    Notes
    -----
    The integrate method uses an scipy "event" that detects low mass and prevents
    overshooting of the ablation model, this event detection mass is the
    `minimum_mass` config variable.
    """

    DEFAULT_CONFIG = {
        "integrate": {
            "minimum_mass": 1e-11 * ureg.kg,
            "max_step_size": 1e-1 * ureg.sec,
            "max_time": 100.0 * ureg.sec,
            "method": "RK45",
        },
        "method_options": {},
    }

    def integrate(self, state, *args, **kwargs):

        def _low_mass(t, y):
            res = y[0] / self.config.getfloat("solver_settings", "minimum_mass") - 1
            # logger.debug(
            #   f'Stopping @ {t:<1.4e} s = {res}: {np.log10(y[0]):1.4e} log10(kg) | {y[0]:1.4e} kg'
            # )
            return res

        _low_mass.terminal = True
        _low_mass.direction = -1

        events = [_low_mass]

        method = self.config["solver_settings"].get("method")
        method_options = {key: val for key, val in self.config.items("method_options")}

        logger.debug(
            f"{self.__class__} integrating IVP:\n- method: "
            f"{method}\n- method-options: {method_options}"
        )
        logger.debug(
            "Config-integrate:\n"
            "\n-- ".join([f"{key}: {val}" for key, val in self.config.items("integrate")])
        )

        ivp_result = solve_ivp(
            fun=lambda t, y: self.rhs(t, y[0], y[1:], *args, **kwargs),
            t_span=(0, self.options["max_time"]),
            y0=state,
            method=self.method,
            max_step=self.options["max_step_size"],
            dense_output=False,
            events=events,
            **method_options,
        )

        logger.debug(f"{self.__class__} IVP integration complete")

        return ivp_result

    @abstractmethod
    def rhs(self, t, m, y, *args, **kwargs):
        pass
