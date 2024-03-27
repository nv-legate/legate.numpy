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

from typing import TYPE_CHECKING, Any

import numpy as np

from .._array.util import add_boilerplate

if TYPE_CHECKING:
    from .._array.util import ndarray


@add_boilerplate("a")
def mean(
    a: ndarray,
    axis: int | tuple[int, ...] | None = None,
    dtype: np.dtype[Any] | None = None,
    out: ndarray | None = None,
    keepdims: bool = False,
    where: ndarray | None = None,
) -> ndarray:
    """

    Compute the arithmetic mean along the specified axis.

    Returns the average of the array elements.  The average is taken over
    the flattened array by default, otherwise over the specified axis.
    `float64` intermediate and return values are used for integer inputs.

    Parameters
    ----------
    a : array_like
        Array containing numbers whose mean is desired. If `a` is not an
        array, a conversion is attempted.
    axis : None or int or tuple[int], optional
        Axis or axes along which the means are computed. The default is to
        compute the mean of the flattened array.

        If this is a tuple of ints, a mean is performed over multiple axes,
        instead of a single axis or all the axes as before.
    dtype : data-type, optional
        Type to use in computing the mean.  For integer inputs, the default
        is `float64`; for floating point inputs, it is the same as the
        input dtype.
    out : ndarray, optional
        Alternate output array in which to place the result.  The default
        is ``None``; if provided, it must have the same shape as the
        expected output, but the type will be cast if necessary.
        See `ufuncs-output-type` for more details.

    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.

        If the default value is passed, then `keepdims` will not be
        passed through to the `mean` method of sub-classes of
        `ndarray`, however any non-default value will be.  If the
        sub-class' method does not implement `keepdims` any
        exceptions will be raised.

    where : array_like of bool, optional
        Elements to include in the mean.

    Returns
    -------
    m : ndarray
        If `out is None`, returns a new array of the same dtype a above
        containing the mean values, otherwise a reference to the output
        array is returned.

    See Also
    --------
    numpy.mean

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return a.mean(
        axis=axis, dtype=dtype, out=out, keepdims=keepdims, where=where
    )


@add_boilerplate("a")
def nanmean(
    a: ndarray,
    axis: int | tuple[int, ...] | None = None,
    dtype: np.dtype[Any] | None = None,
    out: ndarray | None = None,
    keepdims: bool = False,
    where: ndarray | None = None,
) -> ndarray:
    """

    Compute the arithmetic mean along the specified axis, ignoring NaNs.

    Returns the average of the array elements.  The average is taken over
    the flattened array by default, otherwise over the specified axis.
    `float64` intermediate and return values are used for integer inputs.

    Parameters
    ----------
    a : array_like
        Array containing numbers whose mean is desired. If `a` is not an
        array, a conversion is attempted.
    axis : None or int or tuple[int], optional
        Axis or axes along which the means are computed. The default is to
        compute the mean of the flattened array.

        If this is a tuple of ints, a mean is performed over multiple axes,
        instead of a single axis or all the axes as before.
    dtype : data-type, optional
        Type to use in computing the mean.  For integer inputs, the default
        is `float64`; for floating point inputs, it is the same as the
        input dtype.
    out : ndarray, optional
        Alternate output array in which to place the result.  The default
        is ``None``; if provided, it must have the same shape as the
        expected output, but the type will be cast if necessary.
        See `ufuncs-output-type` for more details.

    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.


    where : array_like of bool, optional
        Elements to include in the mean.

    Returns
    -------
    m : ndarray
        If `out is None`, returns a new array of the same dtype as a above
        containing the mean values, otherwise a reference to the output
        array is returned.

    See Also
    --------
    numpy.nanmean

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return a._nanmean(
        axis=axis, dtype=dtype, out=out, keepdims=keepdims, where=where
    )


@add_boilerplate("a")
def var(
    a: ndarray,
    axis: int | tuple[int, ...] | None = None,
    dtype: np.dtype[Any] | None = None,
    out: ndarray | None = None,
    ddof: int = 0,
    keepdims: bool = False,
    *,
    where: ndarray | None = None,
) -> ndarray:
    """
    Compute the variance along the specified axis.

    Returns the variance of the array elements, a measure of the spread of
    a distribution. The variance is computed for the flattened array
    by default, otherwise over the specified axis.

    Parameters
    ----------
    a : array_like
        Array containing numbers whose variance is desired. If `a` is not an
        array, a conversion is attempted.
    axis : None or int or tuple[int], optional
        Axis or axes along which the variance is computed. The default is to
        compute the variance of the flattened array.

        If this is a tuple of ints, a variance is performed over multiple axes,
        instead of a single axis or all the axes as before.
    dtype : data-type, optional
        Type to use in computing the variance. For arrays of integer type
        the default is float64; for arrays of float types
        it is the same as the array type.
    out : ndarray, optional
        Alternate output array in which to place the result. It must have the
        same shape as the expected output, but the type is cast if necessary.
    ddof : int, optional
        “Delta Degrees of Freedom”: the divisor used in the calculation is
        N - ddof, where N represents the number of elements. By default
        ddof is zero.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.
    where : array_like of bool, optional
        A boolean array which is broadcasted to match the dimensions of array,
        and selects elements to include in the reduction.

    Returns
    -------
    m : ndarray, see dtype parameter above
        If `out=None`, returns a new array of the same dtype as above
        containing the variance values, otherwise a reference to the output
        array is returned.

    See Also
    --------
    numpy.var

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return a.var(
        axis=axis,
        dtype=dtype,
        out=out,
        ddof=ddof,
        keepdims=keepdims,
        where=where,
    )
