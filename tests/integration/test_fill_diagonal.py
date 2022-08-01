# Copyright 2022 NVIDIA Corporation
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
from utils.generators import mk_seq_array

import cunumeric as num
from legate.core import LEGATE_MAX_DIM


@pytest.mark.parametrize("ndim", range(2, LEGATE_MAX_DIM + 1))
def test_fill_diagonal(ndim):
    shape = (5,) * ndim
    np_array = mk_seq_array(np, shape)
    num_array = num.array(np_array)
    np_res = np.fill_diagonal(np_array, 10)
    num_res = num.fill_diagonal(num_array, 10)
    assert np.array_equal(np_res, num_res)

    # values is an array:
    np_values = mk_seq_array(np, 5) * 10
    num_values = num.array(np_values)
    np_res = np.fill_diagonal(np_array, np_values)
    num_res = num.fill_diagonal(num_array, num_values)
    assert np.array_equal(np_res, num_res)

    # values is array that needs to be broadcasted:
    np_values = mk_seq_array(np, 3) * 100
    num_values = num.array(np_values)
    np_res = np.fill_diagonal(np_array, np_values)
    num_res = num.fill_diagonal(num_array, num_values)
    assert np.array_equal(np_res, num_res)

    # values are 2d that need to be broadcasted
    np_values = mk_seq_array(np, (2, 2)) * 100
    num_values = num.array(np_values)
    np_res = np.fill_diagonal(np_array, np_values)
    num_res = num.fill_diagonal(num_array, num_values)
    assert np.array_equal(np_res, num_res)

    # values are 3d that need to be broadcasted
    np_values = mk_seq_array(np, (2, 2, 2)) * 100
    num_values = num.array(np_values)
    np_res = np.fill_diagonal(np_array, np_values)
    num_res = num.fill_diagonal(num_array, num_values)
    assert np.array_equal(np_res, num_res)

    # values are too long
    np_values = mk_seq_array(np, (2, 2, 6)) * 100
    num_values = num.array(np_values)
    np_res = np.fill_diagonal(np_array, np_values)
    num_res = num.fill_diagonal(num_array, num_values)
    assert np.array_equal(np_res, num_res)


SHAPES = [(10, 100), (1, 100), (10, 11)]
VALUE_SHAPES = [(5,), (5, 5), (2,)]


@pytest.mark.parametrize("shape, vshape", zip(SHAPES, VALUE_SHAPES), ids=str)
def test_tall_matrices(shape, vshape):
    a_np = np.random.rand(*shape)
    v_np = np.random.rand(*vshape)
    a_num = num.array(a_np)
    v_num = num.array(v_np)
    res_np = np.fill_diagonal(a_np, v_np)
    res_num = num.fill_diagonal(a_num, v_num)
    assert np.array_equal(res_np, res_num)

    res_np = np.fill_diagonal(a_np, v_np, wrap=True)
    res_num = num.fill_diagonal(a_num, v_num, wrap=True)
    assert np.array_equal(res_np, res_num)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))