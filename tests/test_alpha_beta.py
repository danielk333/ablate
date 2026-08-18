import numpy as np
import pytest

from metablate.physics import alpha_beta


def test_alpha_ballistic_coefficient_roundtrip():
    alpha = 12.0
    sea_level_rho = 1.225
    scale_height = 7160.0
    elevation = np.deg2rad(35.0)

    ballistic = alpha_beta.ballistic_coefficient_from_alpha(
        alpha,
        sea_level_rho,
        scale_height,
        elevation,
    )
    recovered = alpha_beta.alpha_from_ballistic_coefficient(
        ballistic,
        sea_level_rho,
        scale_height,
        elevation,
    )

    assert recovered == pytest.approx(alpha, rel=1e-12)


def test_mass_velocity_roundtrip():
    initial_mass = 2.5e-5
    initial_velocity = 41.0e3
    beta = 2.0
    mu = 0.4
    velocities = np.array([0.55, 0.7, 0.9]) * initial_velocity

    mass = alpha_beta.mass_direct(velocities, initial_mass, beta, mu, initial_velocity=initial_velocity)
    recovered_velocity = alpha_beta.velocity_direct(
        mass,
        beta,
        initial_velocity,
        mu,
        initial_mass=initial_mass,
    )

    np.testing.assert_allclose(recovered_velocity, velocities, rtol=1e-12)
    assert np.all(mass <= initial_mass)


def test_initial_mass_direct_inverts_alpha_direct_for_shape_factor():
    mass = 1.0e-6
    density = 1000.0
    radius = ((mass / density) / (4 * np.pi / 3)) ** (1.0 / 3.0)
    cross_section = np.pi * radius**2
    shape_factor = alpha_beta.shape_factor_direct(cross_section, mass, density)

    alpha = alpha_beta.alpha_direct(
        aerodynamic_cd=1.0,
        sea_level_rho=1.225,
        atmospheric_scale_height=7160.0,
        initial_cross_section=cross_section,
        initial_mass=mass,
        radiant_local_elevation=np.deg2rad(45.0),
    )
    recovered_mass = alpha_beta.initial_mass_direct(
        alpha,
        aerodynamic_cd=1.0,
        sea_level_rho=1.225,
        atmospheric_scale_height=7160.0,
        radiant_local_elevation=np.deg2rad(45.0),
        bulk_density=density,
        shape_factor=shape_factor,
    )

    assert recovered_mass == pytest.approx(mass, rel=1e-12)


def test_height_velocity_roundtrip_with_root_estimate():
    alpha = 20.0
    beta = 1.5
    normalized_velocity = np.array([0.55, 0.7, 0.85])
    height = alpha_beta.norm_height_direct(normalized_velocity, alpha, beta)

    recovered = alpha_beta.norm_velocity_estimate(height, alpha, beta)

    np.testing.assert_allclose(recovered, normalized_velocity, rtol=1e-5, atol=1e-5)
