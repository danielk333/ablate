#!/usr/bin/env python

import mutationpp as mpp
import numpy as np
from cycler import cycler
import matplotlib.pyplot as plt

colors = ["r", "g", "b", "k"]
linestyles = ["-", "--", ":", "-.", "-"]
style_cycler = cycler(linestyle=linestyles) * cycler(color=colors)

myMixtureOptions = mpp.MixtureOptions("air_11")
print(f"Mixture path: {myMixtureOptions.getSource()}")
print(f"Mixture species: {myMixtureOptions.getSpeciesDescriptor()}")

myMixtureOptions.setStateModel("Equil")
mix = mpp.Mixture(myMixtureOptions)

ne = mix.nElements()
ns = mix.nSpecies()
print(f"Mixture Elements: {ne}")
print(f"Mixture Species: {ns}")

Tin = 300.0
Tout = 15000.0
temperatures = np.linspace(Tin, Tout, 1000)
pressure = 1000.0


def get_species_descriptor(mixture):
    spec = np.empty((len(temperatures), ns), dtype=np.float64)
    for ind, T in enumerate(temperatures):
        mixture.setState(pressure, T, 1)
        spec[ind, :] = mixture.X()
    return spec


print("Average diffusion coefficients:\n", mix.averageDiffusionCoeffs())
print("Number of energy equations:\n", mix.nEnergyEqns())
print("Species molecular weight:\n", mix.speciesMw())
print("Element matrix:\n", mix.elementMatrix())


species_descriptor = get_species_descriptor(mix)

fig, ax = plt.subplots(layout="tight")
ax.set_prop_cycle(style_cycler)
for ind in range(ns):
    ax.plot(temperatures, species_descriptor[:, ind], label=mix.speciesName(ind))
ax.set_ylabel("Mole fraction [1]")
ax.set_xlabel("Temperature [K]")
ax.legend()

mix.addComposition("N2:1.0, O2:0.0", True)
species_descriptor = get_species_descriptor(mix)

fig, ax = plt.subplots(layout="tight")
ax.set_prop_cycle(style_cycler)
for ind in range(ns):
    ax.plot(temperatures, species_descriptor[:, ind], label=mix.speciesName(ind))
ax.set_ylabel("Mole fraction [1]")
ax.set_xlabel("Temperature [K]")
ax.legend()

plt.show()
