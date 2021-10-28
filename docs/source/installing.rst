.. _installing:

Installation
============

The recommended install command is:

.. code-block:: bash

    pip install ablate


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
