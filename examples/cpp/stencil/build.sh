#!/bin/bash

# Copyright 2023 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cunumeric_root=`python -c 'import cunumeric.install_info as i; from pathlib import Path; print(Path(i.libpath).parent.resolve())'`
echo "Using cuNumeric at $cunumeric_root"
cmake -S . -B build -D cunumeric_ROOT="$cunumeric_root" -D CMAKE_BUILD_TYPE=Debug
cmake --build build --parallel 8
