import setuptools

with open("README", "r") as fh:
    long_description = fh.read()

with open('requirements', 'r') as fh:
    pip_req = fh.read().split('\n')
    pip_req = [x.strip() for x in pip_req]

setuptools.setup(
    name='ablate',
    version='0.1',
    scripts=[''] ,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/javatechy/dokr",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU-GPLv3",
        "Operating System :: OS Independent",
    ],
    install_requires=pip_req,
    packages=setuptools.find_packages(),
    scripts=['say_hello.py'],
    package_data={
        # If any package contains *.txt or *.rst files, include them:
        '': ['*.txt', '*.rst'],
    },
    # metadata to display on PyPI
    author="Daniel Kastinen",
    author_email="daniel.kastinen@irf.se",
    description="A collection of Meteoroid Ablation Models",
    license="GNU-GPLv3",
)
