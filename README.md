# Meteoroid Ablation Models

## Install

```bash
    pip install ablate
```

## Minimal micrometeoroid ablation model example

The example below runs the built-in `AtmPymsis` atmosphere with the
`KeroSzasz2008` ablation model and recreates a compact sweep over entry
elevation angle and velocity.

![Kero-Szasz ablation sweep](docs/assets/kero_figure1_example.png)

```bash
python docs/examples/ablation_sweep.py --output docs/assets/kero_figure1_example.png
```

The example setup is:

```python
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
```

### matplotlib backends

on arch to get matplotlib to work

`pip install Qt5Agg`

and add a `~/.config/matplotlib/matplotlibrc` with `backend:qt5agg`.

see `https://matplotlib.org/stable/tutorials/introductory/usage.html#backends`

### msise00 atmospheric model

make sure cmake is installed! `sudo pacman -S cmake` and also gfortran `sudo pacman -S gcc gcc-fortran` and the basic libs `sudo pacman -S base-devel`

if compilation fails use the enviormennt variables, for bash `export FC="/usr/bin/gfortran"` in fish `set -x FC "/usr/bin/gfortran"`. If cmake still tries to use gcc for compilation, make sure to remove the build folder under `my_env/lib/pythonX.X/site-packages/msise00/build` so that no cached files override the enviornment variables.


```bash
    mkdir /my/env/msise00_source
    cd /my/env/msise00_source
    git clone https://github.com/scivision/msise00
    cd msise00
    pip install -e .
    python -c "import msise00; msise00.base.build()"
```

or

```bash
pip install git+git://github.com/space-physics/msise00.git@main
python -c "import msise00; msise00.base.build()"
```
