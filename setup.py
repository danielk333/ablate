import os
import setuptools
from setuptools.command.develop import develop
from setuptools.command.install import install
import subprocess

class PostDevelopCommand(develop):
    """Post-installation for development mode."""
    

    def run(self):

        develop.run(self)


class PostInstallCommand(install):
    """Post-installation for installation mode."""
    

    def run(self):

        install.run(self)


with open('README.rst', 'r') as fh:
    long_description = fh.read()


with open('requirements', 'r') as fh:
    pip_req = fh.read().split('\n')
    pip_req = [x.strip() for x in pip_req if len(x.strip()) > 0]


setuptools.setup(
    name='ablate',
    version='0.0.1',
    long_description=long_description,
    url='https://gitlab.irf.se/danielk/ablation_models',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU-GPLv3',
        'Operating System :: OS Independent',
    ],
    install_requires=pip_req,
    packages=setuptools.find_packages(),
    package_data={
        '': ['*.txt', '*.rst'],
    },
    # metadata to display on PyPI
    author='Daniel Kastinen',
    author_email='daniel.kastinen@irf.se',
    description='A collection of Meteoroid Ablation Models',
    license='GNU-GPLv3',
    cmdclass={
        'develop': PostDevelopCommand,
        'install': PostInstallCommand,
    },
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
)
