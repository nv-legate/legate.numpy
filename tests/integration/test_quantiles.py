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
from legate.core import LEGATE_MAX_DIM
from utils.comparisons import allclose

import cunumeric as num

ALL_METHODS = (
    "inverted_cdf",
    "averaged_inverted_cdf",
    "closest_observation",
    "interpolated_inverted_cdf",
    "hazen",
    "weibull",
    "linear",
    "median_unbiased",
    "normal_unbiased",
    "lower",
    "higher",
    "midpoint",
    "nearest",
)


@pytest.mark.parametrize("str_method", ALL_METHODS)
@pytest.mark.parametrize("axes", (0, 1, (0, 1), (0, 2)))
@pytest.mark.parametrize(
    "qin_arr", (0.5, [0.001, 0.37, 0.42, 0.67, 0.83, 0.99, 0.39, 0.49, 0.5])
)
@pytest.mark.parametrize("keepdims", (False, True))
@pytest.mark.parametrize("overwrite_input", (False, True))
def test_multi_axes(str_method, axes, qin_arr, keepdims, overwrite_input):
    eps = 1.0e-8
    arr = np.ndarray(
        shape=(2, 3, 4),
        buffer=np.array(
            [
                1,
                2,
                2,
                40,
                1,
                1,
                2,
                1,
                0,
                10,
                3,
                3,
                40,
                15,
                3,
                7,
                5,
                4,
                7,
                3,
                5,
                1,
                0,
                9,
            ]
        ),
        dtype=int,
    )

    if num.isscalar(qin_arr):
        qs_arr = qin_arr
    else:
        qs_arr = np.array(qin_arr)

    # cunumeric:
    # print("cunumeric axis = %d:"%(axis))
    q_out = num.quantile(
        arr,
        qs_arr,
        axis=axes,
        method=str_method,
        keepdims=keepdims,
        overwrite_input=overwrite_input,
    )
    # print(q_out)

    # np:
    # print("numpy axis = %d:"%(axis))
    np_q_out = np.quantile(
        arr,
        qs_arr,
        axis=axes,
        method=str_method,
        keepdims=keepdims,
        overwrite_input=overwrite_input,
    )
    # print(np_q_out)

    assert q_out.shape == np_q_out.shape
    assert q_out.dtype == np_q_out.dtype

    assert allclose(np_q_out, q_out, atol=eps)


@pytest.mark.parametrize("str_method", ALL_METHODS)
@pytest.mark.parametrize(
    "ls_in",
    (
        [[1.0, 0.13, 2.11], [1.9, 9.2, 0.17]],
        [
            [1, 1, 0],
            [2, 1, 10],
            [2, 2, 3],
            [40, 1, 3],
            [40, 5, 5],
            [15, 4, 1],
            [3, 7, 0],
            [7, 3, 9],
        ],
    ),
)
@pytest.mark.parametrize("axes", (0, 1))
@pytest.mark.parametrize("keepdims", (False, True))
def test_nd_quantile(str_method, ls_in, axes, keepdims):
    eps = 1.0e-8

    arr = np.array(ls_in)

    qs_arr = np.ndarray(
        shape=(2, 4),
        buffer=np.array([0.001, 0.37, 0.42, 0.5, 0.67, 0.83, 0.99, 0.39]).data,
    )

    # cunumeric:
    # print("cunumeric axis = %d:"%(axis))
    q_out = num.quantile(
        arr, qs_arr, axis=axes, method=str_method, keepdims=keepdims
    )
    # print(q_out)

    # np:
    # print("numpy axis = %d:"%(axis))
    np_q_out = np.quantile(
        arr, qs_arr, axis=axes, method=str_method, keepdims=keepdims
    )
    # print(np_q_out)

    assert q_out.shape == np_q_out.shape
    assert q_out.dtype == np_q_out.dtype

    assert allclose(np_q_out, q_out, atol=eps)


@pytest.mark.parametrize("str_method", ALL_METHODS)
@pytest.mark.parametrize("axes", (0, 1))
@pytest.mark.parametrize(
    "qs_arr",
    (
        0.5,
        np.ndarray(
            shape=(5, 6), buffer=np.array([x / 30.0 for x in range(0, 30)])
        ),
    ),
)
@pytest.mark.parametrize("keepdims", (False, True))
def test_quantiles_w_output(str_method, axes, qs_arr, keepdims):
    eps = 1.0e-8
    original_shape = (2, 3, 4)
    arr = np.ndarray(
        shape=original_shape,
        buffer=np.array(
            [
                1,
                2,
                2,
                40,
                1,
                1,
                2,
                1,
                0,
                10,
                3,
                3,
                40,
                15,
                3,
                7,
                5,
                4,
                7,
                3,
                5,
                1,
                0,
                9,
            ]
        ),
        dtype=float,
    )

    # cannot currently run tests with LEGATE_MAX_DIM >= 5
    # (see https://github.com/nv-legate/legate.core/issues/318)
    #
    if (
        (keepdims is True)
        and (num.isscalar(qs_arr) is False)
        and (len(qs_arr.shape) > 1)
        and (LEGATE_MAX_DIM < 5)
    ):
        keepdims = False  # reset keepdims, else len(result.shape)>4

    if keepdims:
        remaining_shape = [
            1 if k == axes else original_shape[k]
            for k in range(0, len(original_shape))
        ]
    else:
        remaining_shape = [
            original_shape[k]
            for k in range(0, len(original_shape))
            if k != axes
        ]

    if num.isscalar(qs_arr):
        q_out = num.zeros(remaining_shape, dtype=float)
        # np_q_out = np.zeros(remaining_shape, dtype=float)
    else:
        q_out = num.zeros((*qs_arr.shape, *remaining_shape), dtype=float)
        # np_q_out = np.zeros((*qs_arr.shape, *remaining_shape), dtype=float)

    # cunumeric:
    # print("cunumeric axis = %d:"%(axis))
    num.quantile(
        arr, qs_arr, axis=axes, out=q_out, method=str_method, keepdims=keepdims
    )
    # print(q_out)

    # np:
    # print("numpy axis = %d:"%(axis))
    # due to numpy bug https://github.com/numpy/numpy/issues/22544
    # out = <not-None> fails with keepdims = True
    #
    np_q_out = np.quantile(
        arr,
        qs_arr,
        axis=axes,
        # out=np_q_out,
        method=str_method,
        keepdims=keepdims,
    )
    # print(np_q_out)

    assert q_out.shape == np_q_out.shape
    assert q_out.dtype == np_q_out.dtype

    assert allclose(np_q_out, q_out, atol=eps)


@pytest.mark.parametrize("str_method", ALL_METHODS)
@pytest.mark.parametrize(
    "qin_arr", (0.5, [0.001, 0.37, 0.42, 0.67, 0.83, 0.99, 0.39, 0.49, 0.5])
)
@pytest.mark.parametrize("keepdims", (False, True))
def test_quantiles_axis_none(str_method, qin_arr, keepdims):
    eps = 1.0e-8
    arr = np.ndarray(
        shape=(2, 3, 4),
        buffer=np.array(
            [
                1,
                2,
                2,
                40,
                1,
                1,
                2,
                1,
                0,
                10,
                3,
                3,
                40,
                15,
                3,
                7,
                5,
                4,
                7,
                3,
                5,
                1,
                0,
                9,
            ]
        ),
        dtype=int,
    )

    if num.isscalar(qin_arr):
        qs_arr = qin_arr
    else:
        qs_arr = np.array(qin_arr)

    # cunumeric:
    # print("cunumeric axis = %d:"%(axis))
    q_out = num.quantile(
        arr,
        qs_arr,
        method=str_method,
        keepdims=keepdims,
    )
    # print(q_out)

    # np:
    # print("numpy axis = %d:"%(axis))
    np_q_out = np.quantile(
        arr,
        qs_arr,
        method=str_method,
        keepdims=keepdims,
    )
    # print(np_q_out)

    assert q_out.shape == np_q_out.shape
    assert q_out.dtype == np_q_out.dtype

    assert allclose(np_q_out, q_out, atol=eps)


@pytest.mark.parametrize("str_method", ALL_METHODS)
@pytest.mark.parametrize(
    "qin_arr", (0.5, [0.001, 0.37, 0.42, 0.67, 0.83, 0.99, 0.39, 0.49, 0.5])
)
@pytest.mark.parametrize("axes", (None, 0))
def test_random_inlined(str_method, qin_arr, axes):
    eps = 1.0e-8
    arr = num.random.random((3, 4, 5))

    q_out = num.quantile(arr, qin_arr, method=str_method, axis=axes)
    np_q_out = np.quantile(arr, qin_arr, method=str_method, axis=axes)

    assert q_out.shape == np_q_out.shape
    assert q_out.dtype == np_q_out.dtype

    assert allclose(np_q_out, q_out, atol=eps)


@pytest.mark.parametrize("str_method", ALL_METHODS)
def test_quantile_at_1(str_method):
    eps = 1.0e-8
    arr = num.arange(4)

    q_out = num.quantile(arr, 1.0, method=str_method)
    np_q_out = np.quantile(arr, 1.0, method=str_method)

    assert q_out.shape == np_q_out.shape
    assert q_out.dtype == np_q_out.dtype

    assert allclose(np_q_out, q_out, atol=eps)


@pytest.mark.parametrize("str_method", ALL_METHODS)
def test_quantile_at_0(str_method):
    eps = 1.0e-8
    arr = num.arange(4)

    q_out = num.quantile(arr, 0.0, method=str_method)
    np_q_out = np.quantile(arr, 0.0, method=str_method)

    assert q_out.shape == np_q_out.shape
    assert q_out.dtype == np_q_out.dtype

    assert allclose(np_q_out, q_out, atol=eps)


@pytest.mark.parametrize("str_method", ALL_METHODS)
@pytest.mark.parametrize(
    "qs_arr",
    (
        0.5,
        np.ndarray(
            shape=(2, 3), buffer=np.array([x / 6.0 for x in range(0, 6)])
        ),
    ),
)
@pytest.mark.parametrize("arr", (3, (3,), [3], (2, 1), [2, 1]))
def test_non_ndarray_input(str_method, qs_arr, arr):
    eps = 1.0e-8

    q_out = num.quantile(arr, qs_arr, method=str_method)
    np_q_out = np.quantile(arr, qs_arr, method=str_method)

    assert q_out.shape == np_q_out.shape
    assert q_out.dtype == np_q_out.dtype

    assert allclose(np_q_out, q_out, atol=eps)


@pytest.mark.parametrize("str_method", ALL_METHODS)
@pytest.mark.parametrize(
    "qs_arr",
    (
        0.5,
        np.ndarray(
            shape=(2, 3), buffer=np.array([x / 6.0 for x in range(0, 6)])
        ),
    ),
)
@pytest.mark.parametrize("keepdims", (False, True))
def test_output_conversion(str_method, qs_arr, keepdims):
    #
    # downcast from float64 to float32, rather than int, until
    # numpy issue: https://github.com/numpy/numpy/issues/22766
    # gets addressed
    #
    eps = 1.0e-8

    arr = num.arange(4, dtype=np.dtype("float64"))

    # get scalars of float32 type:
    #
    num_scalar_out = np.float32(0)
    np_scalar_out = np.float32(0)

    # force downcast (`int` fails due to 22766):
    #
    if num.isscalar(qs_arr):
        q_out = num_scalar_out
        np_q_out = np_scalar_out
    else:
        q_out = num.zeros(qs_arr.shape, dtype=np.dtype("float32"))
        np_q_out = np.zeros(qs_arr.shape, dtype=np.dtype("float32"))

    # temporarily reset keepdims=False due to
    # numpy bug https://github.com/numpy/numpy/issues/22544
    # may interfere with checking proper functionality
    #
    keepdims = False
    num.quantile(arr, qs_arr, method=str_method, keepdims=keepdims, out=q_out)

    np.quantile(
        arr, qs_arr, method=str_method, keepdims=keepdims, out=np_q_out
    )

    if not num.isscalar(q_out):
        assert q_out.shape == np_q_out.shape
        assert q_out.dtype == np_q_out.dtype
        assert allclose(np_q_out, q_out, atol=eps)
    else:
        assert abs(q_out - np_q_out) < eps


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
