.. _usage:

Usage
=====

Running cuNumeric programs
--------------------------

Using cuNumeric as a replacement for NumPy is simple. Replace your NumPy import
statement with cuNumeric:

.. code-block:: python

  import numpy as np

becomes

.. code-block:: python

  import cunumeric as np

Then, run the application like you usually do. For example, if you had a script
``main.py`` written in NumPy that adds two vectors,

.. code-block:: python

    import numpy as np
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([4.0, 3.0, 2.0, 1.0])
    z = x + y
    print(z)

change the import statement to use cuNumeric like below,

.. code-block:: python

    import cunumeric as np
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([4.0, 3.0, 2.0, 1.0])
    z = x + y
    print(z)

And run the program, like this

.. code-block:: sh

    python main.py

By default, this command will use 4 CPUs to run the program, but is
configurable through the LEGATE_CONFIG environment variable. For
example, to use 2 CPUs instead, run the following

.. code-block:: sh

    LEGATE_CONFIG="--cpus 2" python main.py

For more information on how resources can be allocated using this
environment variable, see `Using LEGATE_CONFIG`_.

.. note::

    Usage of standard Python is intended as a quick on-ramp for users to try
    out cuNumeric more easily. Several legate command line configuration
    options, especially for multi-node execution, are not available when
    running programs with standard Python. See the output of ``legate --help``
    for more details.

To fully utilize the power of cuNumeric and overcome these restrictions, we
recommend requesting resource allocation using Legate.

Resource allocation
-------------------

Legate allows you to prescribe the resources required to successfully execute
your application. Applications can be run on three different types of
processors, also known as task variants: CPU, OMP, and GPU. The OMP variant
will use OpenMP threads to parallelize your application while the CPU variant
will use individual processes per processor. In addition to the number or
processors, you can also specify the amount of memory required for your
application on each of these processors.

Check the relevant command line arguments to legate and their default values
before using them. In summary, if you want to change the number of processors,
make sure to check out the following arguments in the documentation for legate:
``--cpus``, ``--omps``, ``--ompthreads``, and ``--gpus``. Similarly, if you
need to change the amount of memory required for your application, check the
following arguments: ``--sysmem``, ``--numamem``, and ``--fbmem``.

Legate reserves a fraction of the requested memory, denoted by
``--eager-alloc-percentage``, to be used eagerly, with the rest used for
deferred allocations. Reducing this typically helps you run larger problems.

If you encounter errors related to resource allocation, check out our
:ref:`faqs` to debug them.

Using legate launcher
~~~~~~~~~~~~~~~~~~~~~

To run the above program using four OpenMP threads using the Legate launcher,
run the following command

.. code-block:: sh

    legate --omps 1 --ompthreads 4 --sysmem 40000 --eager-alloc-percentage 10 ./main.py <main.py options>

This will use one OpenMP group and two OpenMP threads to parallelize the
application. We defer discussions on changing the OpenMP group to a later
section.

To run on 8 CPUs and use 40GB of system memory with 10% of that memory reserved
for eager allocations, use the following command:

.. code-block:: sh

    legate --cpus 8 --sysmem 40000 --eager-alloc-percentage 10 ./main.py <main.py options>

To run on multiple GPUs and use 40GB of framebuffer memory per GPU with 10%
of that memory reserved for eager allocations, use the following command:

.. code-block:: sh

    legate --gpus 2 --fbmem 40000 --eager-alloc-percentage 10 ./main.py <main.py options>

Using LEGATE_CONFIG
~~~~~~~~~~~~~~~~~~~

All of the above commands can also be passed through the environment variable
``LEGATE_CONFIG`` as shown below:

.. code-block:: sh

    LEGATE_CONFIG="--omps 1 --ompthreads 4 --sysmem 40000 --eager-alloc-percentage 10" legate main.py <main.py options>

.. code-block:: sh

    LEGATE_CONFIG="--cpus 8 --sysmem 40000 --eager-alloc-percentage 10" legate main.py <main.py options>

.. code-block:: sh

    LEGATE_CONFIG="--gpus 2 --fbmem 40000 --eager-alloc-percentage 10" legate main.py <main.py options>

Using the environment variable might be useful for users using the same set of
resources for their runs where they can just set the environment variable once
and use ``legate main.py`` for all subsequent runs.
