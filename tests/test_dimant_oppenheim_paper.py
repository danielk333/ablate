"""MetAblate's physical density API compared with the paper's Figure 3 curves."""

import warnings

import numpy as np
import pytest

from metablate.models import dimant_oppenheim_2017 as dimant


# Figure 3: theta = 0, pi/8, ..., pi; R/lambda_T = .05, .1, .3, 1, 3, 5.
# The caption's M normalization (Eq. 17) is pi*ne/n_star, not ne/n_star.
# These fixed values were calculated independently from Eqs. 45-48 using
# a scalar, combined quadrature, not either MetAblate density function:
# https://github.com/jvierine/sabmod/blob/bb2388d4f45dba6aaa0e944f29bdf686e54a30d4/dimant_oppenheim.py
# Paper: https://doi.org/10.1002/2017JA023963
# They are equation-derived reference values, not digitized image pixels.
PAPER_FIGURE3 = np.array([
    [149.66131265199, 71.509139008418, 20.846028901557, 4.5463564043972, .84478217707980, .34118798330877],
    [141.50910182285, 67.539055086792, 19.628383122772, 4.2546936674336, .78466022621577, .31598633181820],
    [118.73453073614, 56.444759944278, 16.223734435290, 3.4377948011007, .61527281826336, .24466787219112],
    [86.476225900630, 40.721236699210, 11.391832148977, 2.2739674903245, .37071318061761, .14048946452297],
    [54.852238838490, 25.305041798455, 6.6547482626067, 1.1323926132662, .12783724417678, .035076419422834],
    [40.466141909064, 18.369500898100, 4.6045252113375, .69348131692896, .058313665447603, .011807321875943],
    [33.718980958278, 15.144137498788, 3.6824265141766, .51743714706625, .038033782199331, .0068942766085549],
    [30.430933044490, 13.577191373172, 3.2423746619345, .43906251247887, .030461079813570, .0053201578045246],
    [29.431169143525, 13.101238611556, 3.1099411496210, .41634700308159, .028442903546872, .0049253396616855],
])


@pytest.mark.parametrize("angle_index", range(9))
def test_physical_density_matches_paper_figure3(angle_index):
    theta = angle_index * np.pi / 8
    parameters = dict(
        total_atmospheric_number_density=1e19,
        meteoroid_velocity=40e3,
        meteoroid_radius=5e-5,
        plasma_source_density=1e20,
        collisional_cross_section=3e-20,
        ionization_probability=.5,
        ablated_thermal_speed=951.,
        atmospheric_species_mass=4.7e-26,
        ablated_species_mass=3.8e-26,
    )
    # Eq. 49 and the coefficient of Eq. 45 evaluated independently.
    lambda_T = 951. / (1e19 * 3e-20 * 40e3)
    n_star = 2 * (5e-5)**2 * 1e20 * 1e19 * 3e-20 * .5 * (1 + 3.8 / 4.7) / (np.sqrt(3) * lambda_T)
    actual = []
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for radius in [.05, .1, .3, 1., 3., 5.]:
            _, _, R, morphology = dimant.plasma_distribution_morphology_grid(
                np.array([radius * np.cos(theta)]),
                np.array([radius * np.sin(theta)]),
                quad_abs_tol=1e-10, pbar=False,
            )
            ne = dimant.plasma_distribution(
                **parameters, base_plasma_distribution_function=morphology, R_lambda=R,
            )
            actual.append(np.pi * ne[0, 0] / n_star)
    np.testing.assert_allclose(actual, PAPER_FIGURE3[angle_index], rtol=1e-7, atol=1e-10)
