import numpy as np
import metablate.models.kero_szasz_2008 as ks
from spacecoords import frames

lat = 67.0
lon = 20.0
alt = 100e3
reference_pos_ecef = frames.geodetic_wgs84_to_ecef(lat, lon, alt, degrees=True)
velocity_dir_ecef = frames.azel_to_ecef(lat, lon, az=10, el=-45, degrees=True)

model = ks.KeroSzasz2008(
    options=ks.KeroSzaszOptions(
        material=mat.cometary,
        sputtering=False,
        minimum_mass=1e-11,
        max_step_size=1e-3,
        start_altitude=150e3,
    ),
)

result = model.run(
    parameters=ks.KeroSzaszInitialState(
        epoch=np.datetime64("2018-06-28T12:45:33"),
        position_ecef=reference_pos_ecef,
        velocity_ecef=velocity_dir_ecef * 60e3,
        mass=1e-6,
    ),
)
