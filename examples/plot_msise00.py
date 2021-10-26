'''
This is my example script
=========================

Docstring for this example
'''

import ablate.atmosphere as atm
import numpy as np
import matplotlib.pyplot as plt



model = atm.NRLMSISE00()

print('NRL MSISE00 species:')
for name, species_data in model.species.items():
    print(f'{name}:{species_data}')

select_species = ['O', 'N2', 'O2']

data1 = model.density(
    time = np.datetime64('2018-07-28'),
    lat = np.array([69.0, 69.1]),
    lon = np.array([12.0, 12.0]),
    alt = np.linspace(89e3, 120e3, num=100),
)
print(data1)


fig, axes = plt.subplots(2,1)

index = dict(lat=0, lon=0, alt_km=slice(None, None), time=0)

axes[0].plot(data1.alt_km.values, data1['Total'][index].values)
axes[0].set_xlabel('Altitude [km]')
axes[0].set_ylabel('Total density [kg/m^3]')



time = np.datetime64('2018-07-28') + np.timedelta64(1, 's')*np.linspace(0,3600*24.0,num=100)
data2 = model.density(
    time = time,
    lat = 69.0,
    lon = 12.0,
    alt = 120e3,
)

index = dict(lat=0, lon=0, alt_km=0, time=slice(None, None))

axes[1].plot(data2.time.values, data2['Total'][:,0,0,0].values)
axes[1].set_xlabel('Time')
axes[1].set_ylabel('Total density [kg/m^3]')

plt.show()