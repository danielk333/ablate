import ablate.atmosphere as atm
import numpy as np


model = atm.NRLMSISE00()
species = ['O', 'N2', 'O2']
data = model.density(
    npt = np.datetime64('2018-07-28', 's'),
    species = species,
    lat = 69,
    lon = 12,
    alt = 89e3,
)

data = data[0]

print('Date: {}'.format(data['date']))
for key in species:
    print(f'Species {key}: {data[key]} [m^-3]')