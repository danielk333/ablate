.. _installing:

Installation
============

The recommended install command is:

.. code-block:: bash

    pip install --install-option test git+https://gitlab.irf.se/danielk/ablation_models

Since this will also automatically run all associated package-tests before completing installation. If one wishes to install without tests simply omit :code:`--install-option test`.

Dependencies
------------

**msise00**

.. code-block:: bash

    mkdir /my/env/msise00_source
    cd /my/env/msise00_source
    git clone https://github.com/scivision/msise00
    cd msise00
    pip install -e .
    python -c "import msise00; msise00.build()"
