# Copyright 2024 NVIDIA Corporation
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
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Iterable, Optional, Union

import numpy as np

from ..array import add_boilerplate
from .array_transpose import moveaxis
from .creation_shape import zeros
from .ssc_sorting import sort

if TYPE_CHECKING:
    from typing import Callable

    import numpy.typing as npt

    from ..array import ndarray


# for the case when axis = tuple (non-singleton)
# reshuffling might have to be done (if tuple is non-consecutive)
# and the src array must be collapsed along that set of axes
#
# args:
#
# arr:    [in] source nd-array on which quantiles are calculated;
# axes_set: [in] tuple or list of axes (indices less than arr dimension);
#
# return: pair: (minimal_index, reshuffled_and_collapsed source array)
def _reshuffle_reshape(
    arr: ndarray, axes_set: Iterable[int]
) -> tuple[int, ndarray]:
    ndim = len(arr.shape)

    sorted_axes = tuple(sorted(axes_set))

    min_dim_index = sorted_axes[0]
    num_axes = len(sorted_axes)
    reshuffled_axes = tuple(range(min_dim_index, min_dim_index + num_axes))

    non_consecutive = sorted_axes != reshuffled_axes
    if non_consecutive:
        arr_shuffled = moveaxis(arr, sorted_axes, reshuffled_axes)
    else:
        arr_shuffled = arr

    # shape_reshuffled = arr_shuffled.shape # debug
    collapsed_shape = np.prod([arr_shuffled.shape[i] for i in reshuffled_axes])

    redimed = tuple(range(0, min_dim_index + 1)) + tuple(
        range(min_dim_index + num_axes, ndim)
    )
    reshaped = tuple(
        [
            collapsed_shape if k == min_dim_index else arr_shuffled.shape[k]
            for k in redimed
        ]
    )

    arr_reshaped = arr_shuffled.reshape(reshaped)
    return (min_dim_index, arr_reshaped)


# account for 0-based indexing
# there's no negative numbers
# arithmetic at this level,
# (pos, k) are always positive!
#
def _floor_i(k: int | float) -> int:
    j = k - 1 if k > 0 else 0
    return int(j)


# Generic rule: if `q` input value falls onto a node, then return that node

# Discontinuous methods:


# q = quantile input \in [0, 1]
# n = sizeof(array)
def _inverted_cdf(q: float, n: int) -> tuple[float, int]:
    pos = q * n
    k = math.floor(pos)

    g = pos - k
    gamma = 1.0 if g > 0 else 0.0

    j = int(k) - 1
    if j < 0:
        return (0.0, 0)
    else:
        return (gamma, j)


def _averaged_inverted_cdf(q: float, n: int) -> tuple[float, int]:
    pos = q * n
    k = math.floor(pos)

    g = pos - k
    gamma = 1.0 if g > 0 else 0.5

    j = int(k) - 1
    if j < 0:
        return (0.0, 0)
    elif j >= n - 1:
        return (1.0, n - 2)
    else:
        return (gamma, j)


def _closest_observation(q: float, n: int) -> tuple[float, int]:
    # p = q*n - 0.5
    # pos = 0 if p < 0 else p

    # weird departure from paper
    # (bug?), but this fixes it:
    # also, j even in original paper
    # applied to 1-based indexing; we have 0-based!
    # numpy impl. doesn't account that the original paper used
    # 1-based indexing, 0-based j is still checked for evennes!
    # (see proof in quantile_policies.py)
    #
    p0 = q * n - 0.5
    p = p0 - 1.0

    pos = 0 if p < 0 else p0
    k = math.floor(pos)

    j = _floor_i(k)
    gamma = 1 if k < pos else (0 if j % 2 == 0 else 1)

    return (gamma, j)


# Continuous methods:


# Parzen method
def _interpolated_inverted_cdf(q: float, n: int) -> tuple[float, int]:
    pos = q * n
    k = math.floor(pos)
    # gamma = pos-k
    # this fixes it:
    #
    gamma = 0.0 if k == 0 else pos - k
    j = _floor_i(k)
    return (gamma, j)


# Hazen method
def _hazen(q: float, n: int) -> tuple[float, int]:
    pos = q * n + 0.5
    k = math.floor(pos)
    # gamma = pos-k
    #
    # this fixes it:
    # (when pos > n: this actually selects the right point,
    #  which is the correct choice, because right = arr[n]
    #  gets invalidated)
    #
    gamma = 0.0 if (pos < 1 or pos > n) else pos - k

    j = _floor_i(k)
    return (gamma, j)


# Weibull method
def _weibull(q: float, n: int) -> tuple[float, int]:
    pos = q * (n + 1)

    k = math.floor(pos)
    # gamma = pos-k
    #
    # this fixes it:
    # (when pos > n: this actually selects the right point,
    #  which is the correct choice, because right = arr[n]
    #  gets invalidated)
    #
    gamma = 0.0 if (pos < 1 or pos > n) else pos - k

    j = _floor_i(k)

    if j >= n:
        j = n - 1

    return (gamma, j)


# Gumbel method
def _linear(q: float, n: int) -> tuple[float, int]:
    pos = q * (n - 1) + 1
    k = math.floor(pos)
    # gamma = pos-k
    #
    # this fixes it:
    # (when pos > n: this actually selects the right point,
    #  which is the correct choice, because right = arr[n]
    #  gets invalidated)
    #
    gamma = 0.0 if (pos < 1 or pos > n) else pos - k

    j = _floor_i(k)
    return (gamma, j)


# Johnson & Kotz method
def _median_unbiased(q: float, n: int) -> tuple[float, int]:
    fract = 1.0 / 3.0
    pos = q * (n + fract) + fract
    k = math.floor(pos)

    # gamma = pos-k
    #
    # this fixes it:
    # (when pos > n: this actually selects the right point,
    #  which is the correct choice, because right = arr[n]
    #  gets invalidated)
    #
    gamma = 0.0 if (pos < 1 or pos > n) else pos - k

    j = _floor_i(k)
    return (gamma, j)


# Blom method
def _normal_unbiased(q: float, n: int) -> tuple[float, int]:
    fract1 = 0.25
    fract2 = 3.0 / 8.0
    pos = q * (n + fract1) + fract2
    k = math.floor(pos)

    # gamma = pos-k
    #
    # this fixes it:
    # (when pos > n: this actually selects the right point,
    #  which is the correct choice, because right = arr[n]
    #  gets invalidated)
    #
    gamma = 0.0 if (pos < 1 or pos > n) else pos - k

    j = _floor_i(k)
    return (gamma, j)


def _lower(q: float, n: int) -> tuple[float, int]:
    gamma = 0.0
    pos = q * (n - 1)
    k = math.floor(pos)

    j = int(k)
    return (gamma, j)


def _higher(q: float, n: int) -> tuple[float, int]:
    pos = q * (n - 1)
    k = math.floor(pos)

    # Generic rule: (k == pos)
    gamma = 0.0 if (pos == 0 or k == pos) else 1.0

    j = int(k)
    return (gamma, j)


def _midpoint(q: float, n: int) -> tuple[float, int]:
    pos = q * (n - 1)
    k = math.floor(pos)

    # Generic rule: (k == pos)
    gamma = 0.0 if (pos == 0 or k == pos) else 0.5

    j = int(k)
    return (gamma, j)


def _nearest(q: float, n: int) -> tuple[float, int]:
    pos = q * (n - 1)

    # k = floor(pos)
    # gamma = 1.0 if pos - k >= 0.5 else 0.0

    k = np.round(pos)
    gamma = 0.0

    j = int(k)
    return (gamma, j)


# args:
#
# arr:      [in] source nd-array on which quantiles are calculated;
#                preccondition: assumed sorted!
# q_arr:    [in] quantile input values nd-array;
# axis:     [in] axis along which quantiles are calculated;
# method:   [in] func(q, n) returning (gamma, j),
#                where = array1D.size;
# keepdims: [in] boolean flag specifying whether collapsed axis
#                should be kept as dim=1;
# to_dtype: [in] dtype to convert the result to;
# qs_all:   [in/out] result pass through or created (returned)
#
def _quantile_impl(
    arr: ndarray,
    q_arr: npt.NDArray[Any],
    axis: Optional[int],
    axes_set: Iterable[int],
    original_shape: tuple[int, ...],
    method: Callable[[float, int], tuple[float, int]],
    keepdims: bool,
    to_dtype: np.dtype[Any],
    qs_all: Optional[ndarray],
) -> ndarray:
    ndims = len(arr.shape)

    if axis is None:
        n = arr.size

        if keepdims:
            remaining_shape = (1,) * len(original_shape)
        else:
            remaining_shape = ()  # only `q_arr` dictates shape;
        # quantile applied to `arr` seen as 1D;
    else:
        n = arr.shape[axis]

        # arr.shape -{axis}; if keepdims use 1 for arr.shape[axis]:
        # (can be empty [])
        #
        if keepdims:
            remaining_shape = tuple(
                1 if k in axes_set else original_shape[k]
                for k in range(0, len(original_shape))
            )
        else:
            remaining_shape = tuple(
                arr.shape[k] for k in range(0, ndims) if k != axis
            )

    # compose qarr.shape with arr.shape:
    #
    # result.shape = (q_arr.shape, arr.shape -{axis}):
    #
    qresult_shape = (*q_arr.shape, *remaining_shape)

    # construct result NdArray, non-flattening approach:
    #
    if qs_all is None:
        qs_all = zeros(qresult_shape, dtype=to_dtype)
    else:
        # implicit conversion from to_dtype to qs_all.dtype assumed
        #
        if qs_all.shape != qresult_shape:
            raise ValueError("wrong shape on output array")

    for index, q in np.ndenumerate(q_arr):
        (gamma, j) = method(q, n)
        (left_pos, right_pos) = (j, j + 1)

        # (N-1) dimensional ndarray of left, right
        # neighbor values:
        #
        # non-flattening approach:
        #
        # extract values at index=left_pos;
        arr_1D_lvals = arr.take(left_pos, axis)
        arr_vals_shape = arr_1D_lvals.shape

        if right_pos >= n:
            # some quantile methods may result in j==(n-1),
            # hence (j+1) could surpass array boundary;
            #
            arr_1D_rvals = zeros(arr_vals_shape, dtype=arr_1D_lvals.dtype)
        else:
            # extract values at index=right_pos;
            arr_1D_rvals = arr.take(right_pos, axis)

        # vectorized for axis != None;
        # (non-flattening approach)
        #
        if len(index) == 0:
            left = (1.0 - gamma) * arr_1D_lvals.reshape(qs_all.shape)
            right = gamma * arr_1D_rvals.reshape(qs_all.shape)
            qs_all[...] = left + right
        else:
            left = (1.0 - gamma) * arr_1D_lvals.reshape(qs_all[index].shape)
            right = gamma * arr_1D_rvals.reshape(qs_all[index].shape)
            qs_all[index] = left + right

    return qs_all


@add_boilerplate("a")
def quantile(
    a: ndarray,
    q: Union[float, Iterable[float], ndarray],
    axis: Union[None, int, tuple[int, ...]] = None,
    out: Optional[ndarray] = None,
    overwrite_input: bool = False,
    method: str = "linear",
    keepdims: bool = False,
) -> ndarray:
    """
    Compute the q-th quantile of the data along the specified axis.

    Parameters
    ----------
    a : array_like
        Input array or object that can be converted to an array.
    q : array_like of float
        Quantile or sequence of quantiles to compute, which must be between
        0 and 1 inclusive.
    axis : {int, tuple of int, None}, optional
        Axis or axes along which the quantiles are computed. The default is
        to compute the quantile(s) along a flattened version of the array.
    out : ndarray, optional
        Alternative output array in which to place the result. It must have
        the same shape as the expected output.
    overwrite_input : bool, optional
        If True, then allow the input array `a` to be modified by
        intermediate calculations, to save memory. In this case, the
        contents of the input `a` after this function completes is
        undefined.
    method : str, optional
        This parameter specifies the method to use for estimating the
        quantile.  The options sorted by their R type
        as summarized in the H&F paper [1]_ are:
        1. 'inverted_cdf'
        2. 'averaged_inverted_cdf'
        3. 'closest_observation'
        4. 'interpolated_inverted_cdf'
        5. 'hazen'
        6. 'weibull'
        7. 'linear'  (default)
        8. 'median_unbiased'
        9. 'normal_unbiased'
        The first three methods are discontinuous.  NumPy further defines the
        following discontinuous variations of the default 'linear' (7.) option:
        * 'lower'
        * 'higher',
        * 'midpoint'
        * 'nearest'
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in
        the result as dimensions with size one. With this option, the
        result will broadcast correctly against the original array `a`.

    Returns
    -------
    quantile : scalar or ndarray
        If `q` is a single quantile and `axis=None`, then the result
        is a scalar. If multiple quantiles are given, first axis of
        the result corresponds to the quantiles. The other axes are
        the axes that remain after the reduction of `a`. If the input
        contains integers or floats smaller than ``float64``, the output
        data-type is ``float64``. Otherwise, the output data-type is the
        same as that of the input. If `out` is specified, that array is
        returned instead.

    Raises
    ------
    TypeError
        If the type of the input is complex.

    See Also
    --------
    numpy.quantile

    Availability
    --------
    Multiple GPUs, Multiple CPUs

    References
    ----------
    .. [1] R. J. Hyndman and Y. Fan,
       "Sample quantiles in statistical packages,"
       The American Statistician, 50(4), pp. 361-365, 1996
    """

    dict_methods = {
        "inverted_cdf": _inverted_cdf,
        "averaged_inverted_cdf": _averaged_inverted_cdf,
        "closest_observation": _closest_observation,
        "interpolated_inverted_cdf": _interpolated_inverted_cdf,
        "hazen": _hazen,
        "weibull": _weibull,
        "linear": _linear,
        "median_unbiased": _median_unbiased,
        "normal_unbiased": _normal_unbiased,
        "lower": _lower,
        "higher": _higher,
        "midpoint": _midpoint,
        "nearest": _nearest,
    }

    real_axis: Optional[int]
    axes_set: Iterable[int] = []
    original_shape = a.shape

    if axis is not None and isinstance(axis, Iterable):
        if len(axis) == 1:
            real_axis = axis[0]
            a_rr = a
        else:
            (real_axis, a_rr) = _reshuffle_reshape(a, axis)
            # What happens with multiple axes and overwrite_input = True ?
            # It seems overwrite_input is reset to False;
            overwrite_input = False
        axes_set = axis
    else:
        real_axis = axis
        a_rr = a
        if real_axis is not None:
            axes_set = [real_axis]

    # covers both array-like and scalar cases:
    #
    q_arr = np.asarray(q)

    # in the future k-sort (partition)
    # might be faster, for now it uses sort
    # arr = partition(arr, k = floor(nq), axis = real_axis)
    # but that would require a k-sort call for each `q`!
    # too expensive for many `q` values...
    # if no axis given then elements are sorted as a 1D array
    #
    if overwrite_input:
        a_rr.sort(axis=real_axis)
        arr = a_rr
    else:
        arr = sort(a_rr, axis=real_axis)

    if arr.dtype.kind == "c":
        raise TypeError("input array cannot be of complex type")

    # return type dependency on arr.dtype:
    #
    # it depends on interpolation method;
    # For discontinuous methods returning either end of the interval within
    # which the quantile falls, or the other; arr.dtype is returned;
    # else, logic below:
    #
    # if is_float(arr_dtype) && (arr.dtype >= dtype('float64')) then
    #    arr.dtype
    # else
    #    dtype('float64')
    #
    # see https://github.com/numpy/numpy/issues/22323
    #
    if method in [
        "inverted_cdf",
        "closest_observation",
        "lower",
        "higher",
        "nearest",
    ]:
        to_dtype = arr.dtype
    else:
        to_dtype = np.dtype("float64")

        # in case dtype("float128") becomes supported:
        #
        # to_dtype = (
        #     arr.dtype
        #     if (arr.dtype == np.dtype("float128"))
        #     else np.dtype("float64")
        # )

    res = _quantile_impl(
        arr,
        q_arr,
        real_axis,
        axes_set,
        original_shape,
        dict_methods[method],
        keepdims,
        to_dtype,
        out,
    )

    if out is not None:
        # out = res.astype(out.dtype) -- conversion done inside impl
        return out
    else:
        return res


@add_boilerplate("a")
def percentile(
    a: ndarray,
    q: Union[float, Iterable[float], ndarray],
    axis: Union[None, int, tuple[int, ...]] = None,
    out: Optional[ndarray] = None,
    overwrite_input: bool = False,
    method: str = "linear",
    keepdims: bool = False,
) -> ndarray:
    """
    Compute the q-th percentile of the data along the specified axis.

    Parameters
    ----------
    a : array_like
        Input array or object that can be converted to an array.
    q : array_like of float
        Percentile or sequence of percentiles to compute, which must be between
        0 and 100 inclusive.
    axis : {int, tuple of int, None}, optional
        Axis or axes along which the percentiles are computed. The default is
        to compute the percentile(s) along a flattened version of the array.
    out : ndarray, optional
        Alternative output array in which to place the result. It must have
        the same shape as the expected output.
    overwrite_input : bool, optional
        If True, then allow the input array `a` to be modified by
        intermediate calculations, to save memory. In this case, the
        contents of the input `a` after this function completes is
        undefined.
    method : str, optional
        This parameter specifies the method to use for estimating the
        percentile.  The options sorted by their R type
        as summarized in the H&F paper [1]_ are:
        1. 'inverted_cdf'
        2. 'averaged_inverted_cdf'
        3. 'closest_observation'
        4. 'interpolated_inverted_cdf'
        5. 'hazen'
        6. 'weibull'
        7. 'linear'  (default)
        8. 'median_unbiased'
        9. 'normal_unbiased'
        The first three methods are discontinuous.  NumPy further defines the
        following discontinuous variations of the default 'linear' (7.) option:
        * 'lower'
        * 'higher',
        * 'midpoint'
        * 'nearest'
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in
        the result as dimensions with size one. With this option, the
        result will broadcast correctly against the original array `a`.

    Returns
    -------
    percentile : scalar or ndarray
        If `q` is a single percentile and `axis=None`, then the result
        is a scalar. If multiple percentiles are given, first axis of
        the result corresponds to the percentiles. The other axes are
        the axes that remain after the reduction of `a`. If the input
        contains integers or floats smaller than ``float64``, the output
        data-type is ``float64``. Otherwise, the output data-type is the
        same as that of the input. If `out` is specified, that array is
        returned instead.

    Raises
    ------
    TypeError
        If the type of the input is complex.

    See Also
    --------
    numpy.percentile

    Availability
    --------
    Multiple GPUs, Multiple CPUs

    References
    ----------
    .. [1] R. J. Hyndman and Y. Fan,
       "Sample quantiles in statistical packages,"
       The American Statistician, 50(4), pp. 361-365, 1996
    """

    q_arr = np.asarray(q)
    q01 = q_arr / 100.0

    return quantile(
        a,
        q01,
        axis,
        out=out,
        overwrite_input=overwrite_input,
        method=method,
        keepdims=keepdims,
    )
