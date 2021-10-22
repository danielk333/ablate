import setuptools
import pathlib
import codecs

HERE = pathlib.Path(__file__).resolve().parents[0]

def get_version(path):
    with codecs.open(path, 'r') as fp:
        for line in fp.read().splitlines():
            if line.startswith('__version__'):
                delim = '"' if '"' in line else "'"
                return line.split(delim)[1]
        else:
            raise RuntimeError("Unable to find version string.")

with open('README.md', 'r') as fh:
    long_description = fh.read()

with open('requirements', 'r') as fh:
    pip_req = fh.read().split('\n')
    pip_req = [x.strip() for x in pip_req if len(x.strip()) > 0]

setuptools.setup(
    name='ablate',
    version=get_version(HERE / 'src' / 'ablate' / 'version.py'),
    long_description=long_description,
    url='https://gitlab.irf.se/danielk/ablation_models',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU-GPLv3',
        'Operating System :: OS Independent',
    ],
    package_dir={
        '':'src',
    },
    install_requires=pip_req,
    packages=setuptools.find_packages(
        where='src',
    ),
    # metadata to display on PyPI
    author='Daniel Kastinen',
    author_email='daniel.kastinen@irf.se',
    description='A collection of Meteoroid Ablation Models',
    license='GNU-GPLv3',
)
