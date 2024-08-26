<!--
Copyright 2024 NVIDIA Corporation

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
[Legion](https://legion.stanford.edu) runtime. Using cuNumeric you can do things like run
[the final example of the Python CFD course](https://github.com/barbagroup/CFDPython/blob/master/lessons/15_Step_12.ipynb)
completely unmodified on 2048 A100 GPUs in a
[DGX SuperPOD](https://www.nvidia.com/en-us/data-center/dgx-superpod/)
and achieve good weak scaling.

<img src="docs/figures/cfd-demo.png" alt="drawing" width="500"/>

cuNumeric works best for programs that have very large arrays of data
that cannot fit in the memory of a single GPU or a single node and need
to span multiple nodes and GPUs. While our implementation of the current
NumPy API is still incomplete, programs that use unimplemented features
will still work (assuming enough memory) by falling back to the
canonical NumPy implementation.

## Installation

cuNumeric is available from [conda](https://docs.conda.io/projects/conda/en/latest/index.html)
on the [legate channel](https://anaconda.org/legate/cunumeric).
See https://docs.nvidia.com/cunumeric/latest/installation.html for
details about different install configurations, or building
cuNumeric from source.

## Documentation

The cuNumeric documentation can be found
[here](https://docs.nvidia.com/cunumeric).

## Contributing

See the discussion on contributing in [CONTRIBUTING.md](CONTRIBUTING.md).

## Contact

For technical questions about Cunumeric and Legate-based tools, please visit
the [community discussion forum](https://github.com/nv-legate/discussion).

If you have other questions, please contact us at legate(at)nvidia.com.
