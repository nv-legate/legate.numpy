# Copyright 2021-2023 NVIDIA Corporation
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

from typing import TYPE_CHECKING, Any, Union

import numpy as np
from legate.core import Scalar
from numpy.core.multiarray import (  # type: ignore [attr-defined]
    normalize_axis_index,
)
from numpy.core.numeric import (  # type: ignore [attr-defined]
    normalize_axis_tuple,
)

from ..config import (
    BinaryOpCode,
    ConvertCode,
    ScanCode,
    UnaryOpCode,
    UnaryRedCode,
)
from ..types import NdShape
from .util import broadcast_where, find_common_type

if TYPE_CHECKING:
    import numpy.typing as npt

    from ..thunk import NumPyThunk
    from .array import ndarray


def get_where_thunk(
    where: Union[None, ndarray], out_shape: NdShape
) -> Union[None, NumPyThunk]:
    from .array import ndarray

    if where is None:
        return where
    if (
        not isinstance(where, ndarray)
        or where.dtype != np.bool_
        or where.shape != out_shape
    ):
        raise RuntimeError("should have converted this earlier")
    return where._thunk


def perform_unary_op(
    op: UnaryOpCode,
    src: ndarray,
    out: Union[Any, None] = None,
    extra_args: Any = None,
    dtype: Union[np.dtype[Any], None] = None,
    out_dtype: Union[np.dtype[Any], None] = None,
) -> ndarray:
    from .array import ndarray

    if out is not None:
        # If the shapes don't match see if we can broadcast
        # This will raise an exception if they can't be broadcast together
        if np.broadcast_shapes(src.shape, out.shape) != out.shape:
            raise ValueError(
                f"non-broadcastable output operand with shape {out.shape} "
                f"doesn't match the broadcast shape {src.shape}"
            )
    else:
        # No output yet, so make one
        out_shape = src.shape

        if dtype is not None:
            out = ndarray(
                shape=out_shape,
                dtype=dtype,
                inputs=(src,),
            )
        elif out_dtype is not None:
            out = ndarray(
                shape=out_shape,
                dtype=out_dtype,
                inputs=(src,),
            )
        else:
            out = ndarray(
                shape=out_shape,
                dtype=src.dtype
                if src.dtype.kind != "c"
                else np.dtype(np.float32)
                if src.dtype == np.dtype(np.complex64)
                else np.dtype(np.float64),
                inputs=(src,),
            )

    if out_dtype is None:
        if out.dtype != src.dtype and not (
            op == UnaryOpCode.ABSOLUTE and src.dtype.kind == "c"
        ):
            temp = ndarray(
                out.shape,
                dtype=src.dtype,
                inputs=(src,),
            )
            temp._thunk.unary_op(
                op,
                src._thunk,
                True,
                extra_args,
            )
            out._thunk.convert(temp._thunk)
        else:
            out._thunk.unary_op(
                op,
                src._thunk,
                True,
                extra_args,
            )
    else:
        if out.dtype != out_dtype:
            temp = ndarray(
                out.shape,
                dtype=out_dtype,
                inputs=(src,),
            )
            temp._thunk.unary_op(
                op,
                src._thunk,
                True,
                extra_args,
            )
            out._thunk.convert(temp._thunk)
        else:
            out._thunk.unary_op(
                op,
                src._thunk,
                True,
                extra_args,
            )
    return out


def perform_unary_reduction(
    op: UnaryRedCode,
    src: ndarray,
    axis: Any = None,
    dtype: Union[np.dtype[Any], None] = None,
    res_dtype: Union[npt.DTypeLike, None] = None,
    out: Union[ndarray, None] = None,
    keepdims: bool = False,
    args: tuple[Scalar, ...] = (),
    initial: Union[int, float, None] = None,
    where: Union[ndarray, None] = None,
) -> ndarray:
    from .array import ndarray

    # When 'res_dtype' is not None, the input and output of the reduction
    # have different types. Such reduction operators don't take a dtype of
    # the accumulator
    if res_dtype is not None:
        assert dtype is None
        dtype = src.dtype
    else:
        # If 'dtype' exists, that determines both the accumulation dtype
        # and the output dtype
        if dtype is not None:
            res_dtype = dtype
        elif out is not None:
            dtype = out.dtype
            res_dtype = out.dtype
        else:
            dtype = src.dtype
            res_dtype = src.dtype

    # TODO: Need to require initial to be given when the array is empty
    #       or a where mask is given.
    if (
        op
        in (
            UnaryRedCode.ARGMAX,
            UnaryRedCode.ARGMIN,
            UnaryRedCode.MAX,
            UnaryRedCode.MIN,
        )
        and src.dtype.kind == "c"
    ):
        raise NotImplementedError(
            "(arg)max/min not supported for complex-type arrays"
        )

    if axis is None:
        axes = tuple(range(src.ndim))
    else:
        axes = normalize_axis_tuple(axis, src.ndim)

    out_shape: NdShape = ()
    for dim in range(src.ndim):
        if dim not in axes:
            out_shape += (src.shape[dim],)
        elif keepdims:
            out_shape += (1,)

    if out is None:
        out = ndarray(shape=out_shape, dtype=res_dtype, inputs=(src, where))
    elif out.shape != out_shape:
        errmsg = (
            f"the output shapes do not match: expected {out_shape} "
            f"but got {out.shape}"
        )
        raise ValueError(errmsg)

    if dtype != src.dtype:
        src = src.astype(dtype)

    if out.dtype == res_dtype:
        result = out
    else:
        result = ndarray(shape=out_shape, dtype=res_dtype, inputs=(src, where))

    where_array = broadcast_where(where, src.shape)
    result._thunk.unary_reduction(
        op,
        src._thunk,
        get_where_thunk(where_array, src.shape),
        axis,
        axes,
        keepdims,
        args,
        initial,
    )

    if result is not out:
        out._thunk.convert(result._thunk)

    return out


def perform_binary_reduction(
    op: BinaryOpCode,
    one: ndarray,
    two: ndarray,
    dtype: np.dtype[Any],
    extra_args: tuple[Scalar, ...] = (),
) -> ndarray:
    from .array import ndarray

    args = (one, two)

    # We only handle bool types here for now
    assert dtype is not None and dtype == np.dtype(np.bool_)

    # Collapsing down to a single value in this case
    # Check to see if we need to broadcast between inputs
    if one.shape != two.shape:
        broadcast = np.broadcast_shapes(one.shape, two.shape)
    else:
        broadcast = None

    common_type = find_common_type(one, two)
    one_thunk = one._maybe_convert(common_type, args)._thunk
    two_thunk = two._maybe_convert(common_type, args)._thunk

    dst = ndarray(shape=(), dtype=dtype, inputs=args)
    dst._thunk.binary_reduction(
        op,
        one_thunk,
        two_thunk,
        broadcast,
        extra_args,
    )
    return dst


def perform_where(mask: ndarray, one: ndarray, two: ndarray) -> ndarray:
    from .array import ndarray

    args = (mask, one, two)

    mask = mask._maybe_convert(np.dtype(np.bool_), args)

    common_type = find_common_type(one, two)
    one = one._maybe_convert(common_type, args)
    two = two._maybe_convert(common_type, args)

    # Compute the output shape
    out_shape = np.broadcast_shapes(mask.shape, one.shape, two.shape)
    out = ndarray(shape=out_shape, dtype=common_type, inputs=args)
    out._thunk.where(mask._thunk, one._thunk, two._thunk)
    return out


def perform_scan(
    op: ScanCode,
    src: ndarray,
    axis: Any = None,
    dtype: Union[npt.DTypeLike, None] = None,
    out: Union[ndarray, None] = None,
    nan_to_identity: bool = False,
) -> ndarray:
    from .array import ndarray

    if src.dtype.kind != "c" and src.dtype.kind != "f":
        nan_to_identity = False

    if dtype is None:
        if out is None:
            if src.dtype.kind == "i":
                # Set dtype to default platform integer
                dtype = np.int_
            else:
                dtype = src.dtype
        else:
            dtype = out.dtype

    # flatten input when axis is None
    if axis is None:
        axis = 0
        src_arr = src.ravel()
    else:
        axis = normalize_axis_index(axis, src.ndim)
        src_arr = src

    if out is not None:
        if dtype != out.dtype:
            # if out array is specified, its type overrules dtype
            dtype = out.dtype
        if out.shape != src_arr.shape:
            raise NotImplementedError(
                "Varried output shape not supported. Output must have "
                "same shape as input (same size if no axis is provided"
            )
    else:
        out = ndarray(shape=src_arr.shape, dtype=dtype)

    if dtype != src_arr.dtype:
        if nan_to_identity:
            if op is ScanCode.SUM:
                nan_op = ConvertCode.SUM
            else:
                nan_op = ConvertCode.PROD
            # If convert is called, it will handle NAN conversion
            nan_to_identity = False
        else:
            nan_op = ConvertCode.NOOP
        # convert input to temporary for type conversion
        temp = ndarray(shape=src_arr.shape, dtype=dtype)
        temp._thunk.convert(src_arr._thunk, nan_op=nan_op)
        src_arr = temp

    out._thunk.scan(
        op,
        src_arr._thunk,
        axis=axis,
        dtype=dtype,
        nan_to_identity=nan_to_identity,
    )
    return out
