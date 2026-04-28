import numpy as np
import matplotlib.pyplot as plt

# import
from metablate.models.dimant_oppenheim_2017 import (
    plasma_distribution_morphology_array,
    plasma_distribution,
    collisional_cross_section_bronshten_1983,
    ionization_probability_Na_vondrak_2008,
    mean_free_path,
    ablated_thermal_speed_bronshten_1983
)


def main():
    # 1) Build (r, theta) grid in units of mean free path (lambda) 
    # r is dimensionless distance in units of lambda
    r = np.linspace(0.001, 0.1, 100)
    # theta = np.linspace(0.5*np.pi, 1.5*np.pi, 50)
    theta = np.array([3.36]) 
    R, TH = np.meshgrid(r, theta)

    # 2) Compute morphology
    Xlam, Ylam, Rlam, ne_morph, t1, t2, t3 = plasma_distribution_morphology_array(
        R, TH, pbar=True
    )

    Xlam, Ylam, Rlam, ne_morph1, t1, t2, t3 = plasma_distribution_morphology_array(
        R, TH, pbar=True, quad_abs_tol = 1e-3
    )

    Xlam, Ylam, Rlam, ne_morph2, t1, t2, t3 = plasma_distribution_morphology_array(
        R, TH, pbar=True, quad_abs_tol = 1e-8
    )


    # 3) Scale to physical electron density using plasma_distribution
    # Put in example-ish parameters 
    total_atm_n = 1e19  # atmospheric number density [m^-3] 
    v = 40000.0  # meteroid velocity [m/s] 
    r_m = 5e-5  # meteoroid radius [m]
    plasma_source_density = 1e18  # example
    R_lambda = 1.0  # dimensionless scaling in DO paper
    ablated_thermal_speed = ablated_thermal_speed_bronshten_1983(meteoroid_surface_temperature = 2000, meteoroid_molecular_mass = 3.8e-26)
    sigma = collisional_cross_section_bronshten_1983(v)  # m^2
    lamb = mean_free_path(
        ablated_thermal_speed,
        total_atm_n,
        sigma,
        v)
    Pion = ionization_probability_Na_vondrak_2008(v)  # dimensionless

    ne = plasma_distribution(
        total_atmospheric_number_density=total_atm_n,
        meteoroid_velocity=v,
        meteoroid_radius=r_m,
        plasma_source_density=plasma_source_density,
        collisional_cross_section=sigma,
        ionization_probability=Pion,
        base_plasma_distribution_function=ne_morph,
        R_lambda=R_lambda,
    )

    # 4) Plot density curves (1D slices)
    # a) along the meteoroid path axis: theta = 0
    # b) perpendicular cut: theta = pi/2
    # both morphology and scaled density

    i0 = np.argmin(np.abs(theta)) # directly behind meteoroid
    i90 = np.argmin(np.abs(theta - (np.pi / 2))) # perpendicular to the meteoroid

  
  
    # figure -1
    plt.figure()
    plt.plot(r, ne_morph1[0, :] - ne_morph2[0, :], label = "ne_morph1")
    plt.title("Components")
    plt.legend()
    plt.show() 
  
    # figure 0
    plt.figure()
    plt.plot(r, t1[0, :], label = "t1")
    plt.plot(r, t2[0, :], label = "t2")
    plt.plot(r, t3[0, :], label = "t3")
    plt.plot(r, ne_morph[0, :], label = "ne_morph")
    plt.title("Components")
    plt.legend()
    plt.show()



    # figure 1
    plt.figure()
    plt.plot(r, ne_morph[i0, :], label="morphology, θ=0")
    plt.plot(r, ne_morph[i90, :], label="morphology, θ=π/2")
    plt.yscale("log")
    plt.xlabel("r / λ")
    plt.ylabel("ne_morph (dimensionless)")
    plt.title("Plasma morphology curves")
    plt.legend()
    plt.grid(True, which="both")
    plt.show()

    # figure 2: with dips
    plt.figure()
    for index in range(len(theta)):
        plt.plot(r, ne_morph[index, :], label=f"{theta[index]}")
    plt.axvline(r_m/lamb, color = "red")
    plt.yscale("log")
    plt.xlabel("r / λ")
    plt.ylabel("ne_morph (dimensionless)")
    plt.title("Plasma morphology curves")
    plt.legend()
    plt.grid(True, which="both")
    plt.show()

    # figure 3
    plt.figure()
    plt.plot(r, ne[i0, :], label="density, θ=0")
    plt.plot(r, ne[i90, :], label="density, θ=π/2")
    plt.yscale("log")
    plt.xlabel("r / λ")
    plt.ylabel(r"ne (m$^{-3}$)")
    plt.title("Scaled electron density curves")
    plt.legend()
    plt.grid(True, which="both")
    plt.show()

    # 5) 2D map (sanity check)
    plt.figure()
    plt.pcolormesh(Xlam, Ylam, ne_morph, shading="auto")
    plt.gca().set_aspect("equal", "box")
    plt.colorbar(label="ne_morph")
    plt.xlabel("X / λ")
    plt.ylabel("Y / λ")
    plt.title("Plasma morphology (2D)")
    plt.show()


if __name__ == "__main__":
    main()

#note: ne_morph dip at r/lambda = 0.06 (ne_morph = 9e-5)

