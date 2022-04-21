# Copyright 2021-2022 NVIDIA Corporation
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
#

import numpy as np
import pytest

import cunumeric as num
from legate.core import LEGATE_MAX_DIM


def test():
    for ndim in range(LEGATE_MAX_DIM + 1):
        shape = (4,) * ndim
        a = num.random.randint(0, 3, size=shape)
        a_np = np.array(a)

        b = np.unique(a)
        b_np = num.unique(a_np)

        assert np.array_equal(b, b_np)


if __name__ == "__main__":
    import sys

    pytest.main(sys.argv)
