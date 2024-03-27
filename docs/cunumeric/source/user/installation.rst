Installation
============

Linux-64 packages for cuNumeric are available `via conda`_:

.. code-block:: sh

  conda install -c nvidia -c conda-forge -c legate cunumeric

The default package contains GPU support, and is compatible with CUDA >= 11.8
(CUDA driver version >= r520), and Volta or later GPU architectures. There are
also CPU-only packages available, and will be automatically selected by conda
when installing on a machine without GPUs.

See :ref:`building cunumeric from source` for instructions on building cuNumeric manually.

.. _via conda: https://anaconda.org/legate/cunumeric