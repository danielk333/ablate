# Meteoroid Ablation Models

## Install

```bash
    pip install ablate
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