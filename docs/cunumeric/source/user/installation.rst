Installation
============

Linux-64 packages for cuNumeric are available from
`conda <https://docs.conda.io/projects/conda/en/latest/index.html>`_
on the `legate channel <https://anaconda.org/legate/cunumeric>`_.
Please make sure you have at least conda version 24.1 installed, then create
a new environment containing cuNumeric:

.. code-block:: sh

  conda install -c nvidia -c conda-forge -c legate cunumeric

Once installed, you can verify the installation by running one of the examples
from the cuNumeric repository, for instance:

.. code-block:: sh

    $ legate examples/black_scholes.py
    Running black scholes on 10K options...
    Elapsed Time: 129.017 ms

Only linux-64 packages are available at the moment.

The default package contains GPU support, and is compatible with CUDA >= 11.4
(CUDA driver version >= r470), and Volta or later GPU architectures. There are
also CPU-only packages available, and will be automatically selected by
``conda`` when installing on a machine without GPUs.

You can force installation of a CPU-only package by requesting it as follows:

.. code-block:: sh

    conda ... cunumeric=*=*_cpu

See :ref:`building cunumeric from source` for instructions on building cuNumeric manually.

.. _from conda: https://anaconda.org/legate/cunumeric