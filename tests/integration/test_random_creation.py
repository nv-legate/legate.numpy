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
from utils.comparisons import allclose

import cunumeric as cn


@pytest.mark.xfail
def test_randn():
    cn.random.seed(42)
    x = cn.random.randn(10)
    np.random.seed(42)
    xn = np.random.randn(10)
    assert allclose(x, xn)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
