Meteoroid Ablation Models
##########################

Install
--------

**BEFORE INSTALL:**

Make sure setuptools >= 41 

.. code-block:: bash

    pip install -U setuptools


To install msise00:

.. code-block:: bash

    mkdir /tmp/ablate_install_msise00
    cd /tmp/ablate_install_msise00
    git clone https://github.com/scivision/msise00
    pip install .
    python -c "import msise00; msise00.build()"


**INSTALL:**

The recommended install command is:

.. code-block:: bash

    pip install --install-option test git+https://gitlab.irf.se/danielk/ablation_models

Since this will also automatically run all associated package-tests before completing installation. If one wishes to install without tests simply omit :code:`--install-option test`.
