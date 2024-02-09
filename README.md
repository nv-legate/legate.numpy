<!--
Copyright 2021-2022 NVIDIA Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

-->

# cuNumeric

cuNumeric is a [Legate](https://github.com/nv-legate/legate.core) library
that aims to provide a distributed and accelerated drop-in replacement for the
[NumPy API](https://numpy.org/doc/stable/reference/) on top of the
[Legion](https://legion.stanford.edu) runtime. Using cuNumeric you do things like run
[the final example of the Python CFD course](https://github.com/barbagroup/CFDPython/blob/master/lessons/15_Step_12.ipynb)
completely unmodified on 2048 A100 GPUs in a [DGX SuperPOD](https://www.nvidia.com/en-us/data-center/dgx-superpod/) and achieve good weak scaling.

<img src="docs/figures/cfd-demo.png" alt="drawing" width="500"/>

cuNumeric works best for programs that have very large arrays of data
that need to span multiple nodes and GPUs. While our implementation of the current
NumPy API is still incomplete, programs that use unimplemented features
will still work (assuming enough memory) by falling back to the
canonical NumPy implementation.

If you have questions, please contact us at legate(at)nvidia.com.

## Installation

Linux-64 packages for cuNumeric are available [via conda](https://anaconda.org/legate/cunumeric):

```sh
conda install -c nvidia -c conda-forge -c legate cunumeric
```

The default package contains GPU support, and is compatible with CUDA >= 11.8
(CUDA driver version >= r520), and Volta or later GPU architectures. There are
also CPU-only packages available, and will be automatically selected by `conda`
when installing on a machine without GPUs.

For details about building cuNumeric from source,
see the [Legate Core](https://github.com/nv-legate/legate.core) documentation

## Usage and Execution

Using cuNumeric as a replacement for NumPy is as simple as replacing:
```python
import numpy as np
```
with:

```python
import cunumeric as np
```

Scripts can be run using the standard `python` interpreter, or using the `legate`
driver script to assist with launching more advanced configurations.
For information about execution with multiple nodes and with GPUs,
see the [Legate Core](https://github.com/nv-legate/legate.core) documentation.


## Contributing

See the discussion in [CONTRIBUTING.md](CONTRIBUTING.md).
