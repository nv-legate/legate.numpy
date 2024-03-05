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

from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np

from .._array.thunk import perform_scan, perform_unary_reduction
from .._array.util import add_boilerplate
from .._ufunc.floating import isnan
from .._ufunc.math import add, multiply
from ..config import ScanCode, UnaryRedCode
from ..settings import settings as cunumeric_settings
from ._unary_red_utils import get_non_nan_unary_red_code
from .indexing import putmask
from .logic_truth import all, any

if TYPE_CHECKING:
    from .._array.array import ndarray


@add_boilerplate("a")
def prod(
    a: ndarray,
    axis: Optional[Union[int, tuple[int, ...]]] = None,
    dtype: Optional[np.dtype[Any]] = None,
    out: Optional[ndarray] = None,
    keepdims: bool = False,
    initial: Optional[Union[int, float]] = None,
    where: Optional[ndarray] = None,
) -> ndarray:
    """

    Return the product of array elements over a given axis.

    Parameters
    ----------
    a : array_like
        Input data.
    axis : None or int or tuple[int], optional
        Axis or axes along which a product is performed.  The default,
        axis=None, will calculate the product of all the elements in the
        input array. If axis is negative it counts from the last to the
        first axis.

        If axis is a tuple of ints, a product is performed on all of the
        axes specified in the tuple instead of a single axis or all the
        axes as before.
    dtype : data-type, optional
        The type of the returned array, as well as of the accumulator in
        which the elements are multiplied.  The dtype of `a` is used by
        default unless `a` has an integer dtype of less precision than the
        default platform integer.  In that case, if `a` is signed then the
        platform integer is used while if `a` is unsigned then an unsigned
        integer of the same precision as the platform integer is used.
    out : ndarray, optional
        Alternative output array in which to place the result. It must have
        the same shape as the expected output, but the type of the output
        values will be cast if necessary.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in the
        result as dimensions with size one. With this option, the result
        will broadcast correctly against the input array.

        If the default value is passed, then `keepdims` will not be
        passed through to the `prod` method of sub-classes of
        `ndarray`, however any non-default value will be.  If the
        sub-class' method does not implement `keepdims` any
        exceptions will be raised.
    initial : scalar, optional
        The starting value for this product. See `~cunumeric.ufunc.reduce` for
        details.

    where : array_like[bool], optional
        Elements to include in the product. See `~cunumeric.ufunc.reduce` for
        details.

    Returns
    -------
    product_along_axis : ndarray, see `dtype` parameter above.
        An array shaped as `a` but with the specified axis removed.
        Returns a reference to `out` if specified.

    See Also
    --------
    numpy.prod

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return multiply.reduce(
        a,
        axis=axis,
        dtype=dtype,
        out=out,
        keepdims=keepdims,
        initial=initial,
        where=where,
    )


@add_boilerplate("a")
def sum(
    a: ndarray,
    axis: Optional[Union[int, tuple[int, ...]]] = None,
    dtype: Optional[np.dtype[Any]] = None,
    out: Optional[ndarray] = None,
    keepdims: bool = False,
    initial: Optional[Union[int, float]] = None,
    where: Optional[ndarray] = None,
) -> ndarray:
    """

    Sum of array elements over a given axis.

    Parameters
    ----------
    a : array_like
        Elements to sum.
    axis : None or int or tuple[int], optional
        Axis or axes along which a sum is performed.  The default,
        axis=None, will sum all of the elements of the input array.  If
        axis is negative it counts from the last to the first axis.

        If axis is a tuple of ints, a sum is performed on all of the axes
        specified in the tuple instead of a single axis or all the axes as
        before.
    dtype : data-type, optional
        The type of the returned array and of the accumulator in which the
        elements are summed.  The dtype of `a` is used by default unless `a`
        has an integer dtype of less precision than the default platform
        integer.  In that case, if `a` is signed then the platform integer
        is used while if `a` is unsigned then an unsigned integer of the
        same precision as the platform integer is used.
    out : ndarray, optional
        Alternative output array in which to place the result. It must have
        the same shape as the expected output, but the type of the output
        values will be cast if necessary.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.

        If the default value is passed, then `keepdims` will not be
        passed through to the `sum` method of sub-classes of
        `ndarray`, however any non-default value will be.  If the
        sub-class' method does not implement `keepdims` any
        exceptions will be raised.
    initial : scalar, optional
        Starting value for the sum. See `~cunumeric.ufunc.reduce` for details.

    where : array_like[bool], optional
        Elements to include in the sum. See `~cunumeric.ufunc.reduce` for
        details.

    Returns
    -------
    sum_along_axis : ndarray
        An array with the same shape as `a`, with the specified
        axis removed.   If `a` is a 0-d array, or if `axis` is None, a scalar
        is returned.  If an output array is specified, a reference to
        `out` is returned.

    See Also
    --------
    numpy.sum

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return add.reduce(
        a,
        axis=axis,
        dtype=dtype,
        out=out,
        keepdims=keepdims,
        initial=initial,
        where=where,
    )


@add_boilerplate("a")
def cumprod(
    a: ndarray,
    axis: Optional[int] = None,
    dtype: Optional[np.dtype[Any]] = None,
    out: Optional[ndarray] = None,
) -> ndarray:
    """
    Return the cumulative product of the elements along a given axis.

    Parameters
    ----------
    a : array_like
        Input array.

    axis : int, optional
        Axis along which the cumulative product is computed. The default (None)
        is to compute the cumprod over the flattened array.

    dtype : dtype, optional
        Type of the returned array and of the accumulator in which the elements
        are multiplied. If dtype is not specified, it defaults to the dtype of
        a, unless a has an integer dtype with a precision less than that of the
        default platform integer. In that case, the default platform integer is
        used.
    out : ndarray, optional
        Alternative output array in which to place the result. It must have the
        same shape and buffer length as the expected output but the type will
        be cast if necessary. See Output type determination for more details.

    Returns
    -------
    cumprod : ndarray
        A new array holding the result is returned unless out is specified, in
        which case a reference to out is returned. The result has the same size
        as a, and the same shape as a if axis is not None or a is a 1-d array.

    See Also
    --------
    numpy.cumprod

    Notes
    -----
    CuNumeric's parallel implementation may yield different results from NumPy
    with floating point and complex types. For example, when boundary values
    such as inf occur they may not propagate as expected. Consider the float32
    array ``[3e+37, 1, 100, 0.01]``. NumPy's cumprod will return a result of
    ``[3e+37, 3e+37, inf, inf]``. However, cuNumeric might internally partition
    the array such that partition 0 has ``[3e+37, 1]``  and partition 1 has
    ``[100, 0.01]``, returning the result ``[3e+37, 3e+37, inf, 3e+37]``.

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return perform_scan(
        ScanCode.PROD,
        a,
        axis=axis,
        dtype=dtype,
        out=out,
        nan_to_identity=False,
    )


@add_boilerplate("a")
def cumsum(
    a: ndarray,
    axis: Optional[int] = None,
    dtype: Optional[np.dtype[Any]] = None,
    out: Optional[ndarray] = None,
) -> ndarray:
    """
    Return the cumulative sum of the elements along a given axis.

    Parameters
    ----------
    a : array_like
        Input array.

    axis : int, optional
        Axis along which the cumulative sum is computed. The default (None) is
        to compute the cumsum over the flattened array.

    dtype : dtype, optional
        Type of the returned array and of the accumulator in which the elements
        are summed. If dtype is not specified, it defaults to the dtype of a,
        unless a has an integer dtype with a precision less than that of the
        default platform integer. In that case, the default platform integer is
        used.
    out : ndarray, optional
        Alternative output array in which to place the result. It must have the
        same shape and buffer length as the expected output but the type will
        be cast if necessary. See Output type determination for more details.

    Returns
    -------
    cumsum : ndarray.
        A new array holding the result is returned unless out is specified, in
        which case a reference to out is returned. The result has the same size
        as a, and the same shape as a if axis is not None or a is a 1-d array.

    See Also
    --------
    numpy.cumsum

    Notes
    -----
    CuNumeric's parallel implementation may yield different results from NumPy
    with floating point and complex types. For example, when boundary values
    such as inf occur they may not propagate as expected. For more explanation
    check cunumeric.cumprod.

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return perform_scan(
        ScanCode.SUM, a, axis=axis, dtype=dtype, out=out, nan_to_identity=False
    )


@add_boilerplate("a")
def nancumprod(
    a: ndarray,
    axis: Optional[int] = None,
    dtype: Optional[np.dtype[Any]] = None,
    out: Optional[ndarray] = None,
) -> ndarray:
    """
    Return the cumulative product of the elements along a given axis treating
    Not a Numbers (NaNs) as one. The cumulative product does not change when
    NaNs are encountered and leading NaNs are replaced by ones.

    Ones are returned for slices that are all-NaN or empty.

    Parameters
    ----------
    a : array_like
        Input array.

    axis : int, optional
        Axis along which the cumulative product is computed. The default (None)
        is to compute the nancumprod over the flattened array.

    dtype : dtype, optional
        Type of the returned array and of the accumulator in which the elements
        are multiplied. If dtype is not specified, it defaults to the dtype of
        a, unless a has an integer dtype with a precision less than that of the
        default platform integer. In that case, the default platform integer is
        used.
    out : ndarray, optional
        Alternative output array in which to place the result. It must have the
        same shape and buffer length as the expected output but the type will
        be cast if necessary. See Output type determination for more details.

    Returns
    -------
    nancumprod : ndarray.
        A new array holding the result is returned unless out is specified, in
        which case a reference to out is returned. The result has the same size
        as a, and the same shape as a if axis is not None or a is a 1-d array.

    See Also
    --------
    numpy.nancumprod

    Notes
    -----
    CuNumeric's parallel implementation may yield different results from NumPy
    with floating point and complex types. For example, when boundary values
    such as inf occur they may not propagate as expected. For more explanation
    check cunumeric.cumprod.

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return perform_scan(
        ScanCode.PROD, a, axis=axis, dtype=dtype, out=out, nan_to_identity=True
    )


@add_boilerplate("a")
def nancumsum(
    a: ndarray,
    axis: Optional[int] = None,
    dtype: Optional[np.dtype[Any]] = None,
    out: Optional[ndarray] = None,
) -> ndarray:
    """
    Return the cumulative sum of the elements along a given axis treating Not a
    Numbers (NaNs) as zero. The cumulative sum does not change when NaNs are
    encountered and leading NaNs are replaced by zeros.

    Zeros are returned for slices that are all-NaN or empty.

    Parameters
    ----------
    a : array_like
        Input array.

    axis : int, optional
        Axis along which the cumulative sum is computed. The default (None) is
        to compute the nancumsum over the flattened array.

    dtype : dtype, optional
        Type of the returned array and of the accumulator in which the elements
        are summed. If dtype is not specified, it defaults to the dtype of a,
        unless a has an integer dtype with a precision less than that of the
        default platform integer. In that case, the default platform integer is
        used.
    out : ndarray, optional
        Alternative output array in which to place the result. It must have the
        same shape and buffer length as the expected output but the type will
        be cast if necessary. See Output type determination for more details.

    Returns
    -------
    nancumsum : ndarray.
        A new array holding the result is returned unless out is specified, in
        which case a reference to out is returned. The result has the same size
        as a, and the same shape as a if axis is not None or a is a 1-d array.

    See Also
    --------
    numpy.nancumsum

    Notes
    -----
    CuNumeric's parallel implementation may yield different results from NumPy
    with floating point and complex types. For example, when boundary values
    such as inf occur they may not propagate as expected. For more explanation
    check cunumeric.cumprod.

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """
    return perform_scan(
        ScanCode.SUM, a, axis=axis, dtype=dtype, out=out, nan_to_identity=True
    )


@add_boilerplate("a")
def nanargmax(
    a: ndarray,
    axis: Any = None,
    out: Union[ndarray, None] = None,
    *,
    keepdims: bool = False,
) -> ndarray:
    """
    Return the indices of the maximum values in the specified axis ignoring
    NaNs. For empty arrays, ValueError is raised. For all-NaN slices,
    ValueError is raised only when CUNUMERIC_NUMPY_COMPATIBILITY
    environment variable is set, otherwise identity is returned.

    Warning: results cannot be trusted if a slice contains only NaNs
    and -Infs.

    Parameters
    ----------
    a : array_like
        Input array.
    axis : int, optional
        By default, the index corresponds to the flattened array, otherwise
        along the specified axis.
    out : ndarray, optional
        If provided, the result will be inserted into this array. It should
        be of the appropriate shape and dtype.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the array.

    Returns
    -------
    index_array : ndarray[int]
        Array of indices into the array. It has the same shape as `a.shape`
        with the dimension along `axis` removed.

    See Also
    --------
    numpy.nanargmin, numpy.nanargmax

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """

    if a.size == 0:
        raise ValueError("attempt to get nanargmax of an empty sequence")

    if cunumeric_settings.numpy_compat() and a.dtype.kind == "f":
        if any(all(isnan(a), axis=axis)):
            raise ValueError("Array/Slice contains only NaNs")

    unary_red_code = get_non_nan_unary_red_code(
        a.dtype.kind, UnaryRedCode.NANARGMAX
    )

    return perform_unary_reduction(
        unary_red_code,
        a,
        axis=axis,
        out=out,
        keepdims=keepdims,
        res_dtype=np.dtype(np.int64),
    )


@add_boilerplate("a")
def nanargmin(
    a: ndarray,
    axis: Any = None,
    out: Union[ndarray, None] = None,
    *,
    keepdims: bool = False,
) -> ndarray:
    """
    Return the indices of the minimum values in the specified axis ignoring
    NaNs. For empty arrays, ValueError is raised. For all-NaN slices,
    ValueError is raised only when CUNUMERIC_NUMPY_COMPATIBILITY
    environment variable is set, otherwise identity is returned.

    Warning: results cannot be trusted if a slice contains only NaNs
    and -Infs.

    Parameters
    ----------
    a : array_like
        Input array.
    axis : int, optional
        By default, the index corresponds to the flattened array, otherwise
        along the specified axis.
    out : ndarray, optional
        If provided, the result will be inserted into this array. It should
        be of the appropriate shape and dtype.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the array.

    Returns
    -------
    index_array : ndarray[int]
        Array of indices into the array. It has the same shape as `a.shape`
        with the dimension along `axis` removed.

    See Also
    --------
    numpy.nanargmin, numpy.nanargmax

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """

    if a.size == 0:
        raise ValueError("attempt to get nanargmin of an empty sequence")

    if cunumeric_settings.numpy_compat() and a.dtype.kind == "f":
        if any(all(isnan(a), axis=axis)):
            raise ValueError("Array/Slice contains only NaNs")

    unary_red_code = get_non_nan_unary_red_code(
        a.dtype.kind, UnaryRedCode.NANARGMIN
    )

    return perform_unary_reduction(
        unary_red_code,
        a,
        axis=axis,
        out=out,
        keepdims=keepdims,
        res_dtype=np.dtype(np.int64),
    )


@add_boilerplate("a")
def nanmin(
    a: ndarray,
    axis: Any = None,
    out: Union[ndarray, None] = None,
    keepdims: bool = False,
    initial: Optional[Union[int, float]] = None,
    where: Optional[ndarray] = None,
) -> ndarray:
    """
    Return minimum of an array or minimum along an axis, ignoring any
    NaNs. When all-NaN slices are encountered, a NaN is returned
    for that slice only when CUNUMERIC_NUMPY_COMPATIBILITY environment
    variable is set, otherwise identity is returned.
    Empty slices will raise a ValueError

    Parameters
    ----------
    a : array_like
        Array containing numbers whose minimum is desired. If a is not an
        array, a conversion is attempted.

    axis : {int, tuple of int, None}, optional
        Axis or axes along which the minimum is computed. The default is to
        compute the minimum of the flattened array.

    out : ndarray, optional
        Alternative output array in which to place the result.  Must
        be of the same shape and buffer length as the expected output.
        See `ufuncs-output-type` for more details.

    keepdims : bool, Optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.

        If the default value is passed, then `keepdims` will not be
        passed through to the `amin` method of sub-classes of
        `ndarray`, however any non-default value will be.  If the
        sub-class' method does not implement `keepdims` any
        exceptions will be raised.

    initial : scalar, optional
        The maximum value of an output element. Must be present to allow
        computation on empty slice. See `~cunumeric.ufunc.reduce` for details.

    where : array_like[bool], optional
        Elements to compare for the minimum. See `~cunumeric.ufunc.reduce`
        for details.

    Returns
    -------
    nanmin : ndarray or scalar
        Minimum of `a`. If `axis` is None, the result is a scalar value.
        If `axis` is given, the result is an array of dimension
        ``a.ndim - 1``.

    Notes
    -----
    CuNumeric's implementation will not raise a Runtime Warning for
    slices with all-NaNs

    See Also
    --------
    numpy.nanmin, numpy.nanmax, numpy.min, numpy.max, numpy.isnan,
    numpy.maximum

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """

    unary_red_code = get_non_nan_unary_red_code(
        a.dtype.kind, UnaryRedCode.NANMIN
    )

    out_array = perform_unary_reduction(
        unary_red_code,
        a,
        axis=axis,
        out=out,
        keepdims=keepdims,
        initial=initial,
        where=where,
    )

    if cunumeric_settings.numpy_compat() and a.dtype.kind == "f":
        all_nan = all(isnan(a), axis=axis, keepdims=keepdims, where=where)
        putmask(out_array, all_nan, np.nan)  # type: ignore

    return out_array


@add_boilerplate("a")
def nanmax(
    a: ndarray,
    axis: Any = None,
    out: Union[ndarray, None] = None,
    keepdims: bool = False,
    initial: Optional[Union[int, float]] = None,
    where: Optional[ndarray] = None,
) -> ndarray:
    """
    Return the maximum of an array or maximum along an axis, ignoring any
    NaNs. When all-NaN slices are encountered, a NaN is returned
    for that slice only when CUNUMERIC_NUMPY_COMPATIBILITY environment
    variable is set, otherwise identity is returned.
    Empty slices will raise a ValueError

    Parameters
    ----------
    a : array_like
        Array containing numbers whose maximum is desired. If a is not
        an array, a conversion is attempted.

    axis : None or int or tuple[int], optional
        Axis or axes along which to operate.  By default, flattened input is
        used.

        If this is a tuple of ints, the maximum is selected over multiple axes,
        instead of a single axis or all the axes as before.

    out : ndarray, optional
        Alternative output array in which to place the result.  Must
        be of the same shape and buffer length as the expected output.
        See `ufuncs-output-type` for more details.

    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.

        If the default value is passed, then `keepdims` will not be
        passed through to the `amax` method of sub-classes of
        `ndarray`, however any non-default value will be.  If the
        sub-class' method does not implement `keepdims` any
        exceptions will be raised.

    initial : scalar, optional
        The minimum value of an output element. Must be present to allow
        computation on empty slice. See `~cunumeric.ufunc.reduce` for details.

    where : array_like[bool], optional
        Elements to compare for the maximum. See `~cunumeric.ufunc.reduce`
        for details.

    Returns
    -------
    nanmax : ndarray or scalar
        An array with the same shape as `a`, with the specified axis
        removed. If `a` is 0-d array, of if axis is None, an ndarray
        scalar is returned. The same dtype as `a` is returned.

    Notes
    -----
    CuNumeric's implementation will not raise a Runtime Warning for
    slices with all-NaNs

    See Also
    --------
    numpy.nanmin, numpy.amax, numpy.isnan, numpy.fmax, numpy.maximum,
    numpy.isfinite

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """

    unary_red_code = get_non_nan_unary_red_code(
        a.dtype.kind, UnaryRedCode.NANMAX
    )

    out_array = perform_unary_reduction(
        unary_red_code,
        a,
        axis=axis,
        out=out,
        keepdims=keepdims,
        initial=initial,
        where=where,
    )

    if cunumeric_settings.numpy_compat() and a.dtype.kind == "f":
        all_nan = all(isnan(a), axis=axis, keepdims=keepdims, where=where)
        putmask(out_array, all_nan, np.nan)  # type: ignore

    return out_array


@add_boilerplate("a")
def nanprod(
    a: ndarray,
    axis: Any = None,
    dtype: Any = None,
    out: Union[ndarray, None] = None,
    keepdims: bool = False,
    initial: Optional[Union[int, float]] = None,
    where: Optional[ndarray] = None,
) -> ndarray:
    """
    Return the product of array elements over a given axis treating
    Not a Numbers (NaNs) as ones.

    One is returned for slices that are all-NaN or empty.

    Parameters
    ----------
    a : array_like
        Input array.
    axis : int, optional
         Axis or axes along which the product is computed. The
         default is to compute the product of the flattened array.
    dtype : data-type, optional
         The type of the returned array and of the accumulator in
         which the elements are summed. By default, the dtype of a
         is used. An exception is when a has an integer type with
         less precision than the platform (u)intp. In that case,
         the default will be either (u)int32 or (u)int64 depending
         on whether the platform is 32 or 64 bits. For inexact
         inputs, dtype must be inexact.
    out : ndarray, optional
        Alternate output array in which to place the result. The
        default is None. If provided, it must have the same shape as
        the expected output, but the type will be cast if necessary.
        See Output type determination for more details. The casting of
        NaN to integer can yield unexpected results.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in the
        result as dimensions with size one. With this option, the result
        will broadcast correctly against the input array.

        If the default value is passed, then `keepdims` will not be
        passed through to the `prod` method of sub-classes of
        `ndarray`, however any non-default value will be.  If the
        sub-class' method does not implement `keepdims` any
        exceptions will be raised.
    initial : scalar, optional
        The starting value for this product. See `~cunumeric.ufunc.reduce` for
        details.
    where : array_like[bool], optional
        Elements to include in the product. See `~cunumeric.ufunc.reduce` for
        details.

    Returns
    -------
    nanprod: ndarray, see `dtype` parameter above.
        A new array holding the result is returned unless out is
        specified, in which case it is returned.

    See Also
    --------
    numpy.prod, numpy.isnan

    Availability
    --------
    Multiple GPUs, Multiple CPUs

    """

    # Note: if the datatype of the input array is int and less
    # than that of the platform int, then a convert task is launched
    # in np.prod to take care of the type casting

    if a.dtype == np.complex128:
        raise NotImplementedError(
            "operation is not supported for complex128 arrays"
        )

    if a.dtype.kind in ("f", "c"):
        unary_red_code = UnaryRedCode.NANPROD
    else:
        unary_red_code = UnaryRedCode.PROD

    return perform_unary_reduction(
        unary_red_code,
        a,
        axis=axis,
        dtype=dtype,
        out=out,
        keepdims=keepdims,
        initial=initial,
        where=where,
    )


@add_boilerplate("a")
def nansum(
    a: ndarray,
    axis: Any = None,
    dtype: Any = None,
    out: Union[ndarray, None] = None,
    keepdims: bool = False,
    initial: Optional[Union[int, float]] = None,
    where: Optional[ndarray] = None,
) -> ndarray:
    """
    Return the sum of array elements over a given axis treating
    Not a Numbers (NaNs) as ones.

    Zero is returned for slices that are all-NaN or empty.

    Parameters
    ----------
    a : array_like
        Array containing numbers whose product is desired. If a is not
        an array, a conversion is attempted.

    axis : None or int or tuple[int], optional
        Axis or axes along which a sum is performed.  The default,
        axis=None, will sum all of the elements of the input array.
        If axis is negative it counts from the last to the first axis.

        If axis is a tuple of ints, a sum is performed on all of the
        axes specified in the tuple instead of a single axis or all
        the axes as before.

    dtype : data-type, optional
        The type of the returned array and of the accumulator in which
        the elements are summed.  The dtype of `a` is used by default
        unless `a` has an integer dtype of less precision than the
        default platform integer.  In that case, if `a` is signed then
        the platform integer is used while if `a` is unsigned then an
        unsigned integer of the same precision as the platform integer
        is used.

    out : ndarray, optional
        Alternative output array in which to place the result. It must
        have the same shape as the expected output, but the type of
        the output values will be cast if necessary.

    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.

    initial : scalar, optional
        Starting value for the sum. See `~cunumeric.ufunc.reduce` for
        details.

    where : array_like[bool], optional
        Elements to include in the sum. See `~cunumeric.ufunc.reduce` for
        details.

    Returns
    -------
    nansum : ndarray, see `dtype` parameter above.
        A new array holding the result is returned unless out is
        specified, in which case it is returned. The result has the
        same size as a, and the same shape as a if axis is not None or
        a is a 1-d array.

    See Also
    --------
    numpy.nansum, numpy.isnan, numpy.isfinite

    Availability
    --------
    Multiple GPUs, Multiple CPUs
    """

    return a._nansum(
        axis=axis,
        dtype=dtype,
        out=out,
        keepdims=keepdims,
        initial=initial,
        where=where,
    )
