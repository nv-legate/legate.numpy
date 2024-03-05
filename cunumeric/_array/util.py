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

import operator
from functools import wraps
from inspect import signature
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Optional,
    ParamSpec,
    Sequence,
    TypeVar,
    Union,
    cast,
)

import numpy as np
from legate.core.utils import OrderedSet

from ..runtime import runtime
from ..types import NdShape

if TYPE_CHECKING:
    import numpy.typing as npt

    from ..types import NdShapeLike
    from .array import ndarray


R = TypeVar("R")
P = ParamSpec("P")


def add_boilerplate(
    *array_params: str,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Adds required boilerplate to the wrapped cunumeric.ndarray or module-level
    function.

    Every time the wrapped function is called, this wrapper will:
    * Convert all specified array-like parameters, plus the special "out"
      parameter (if present), to cuNumeric ndarrays.
    * Convert the special "where" parameter (if present) to a valid predicate.
    """
    keys = OrderedSet(array_params)
    assert len(keys) == len(array_params)

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        assert not hasattr(
            func, "__wrapped__"
        ), "this decorator must be the innermost"

        # For each parameter specified by name, also consider the case where
        # it's passed as a positional parameter.
        indices: OrderedSet[int] = OrderedSet()
        where_idx: Optional[int] = None
        out_idx: Optional[int] = None
        params = signature(func).parameters
        extra = keys - OrderedSet(params)
        assert len(extra) == 0, f"unknown parameter(s): {extra}"
        for idx, param in enumerate(params):
            if param == "where":
                where_idx = idx
            elif param == "out":
                out_idx = idx
            elif param in keys:
                indices.add(idx)

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> R:
            assert (where_idx is None or len(args) <= where_idx) and (
                out_idx is None or len(args) <= out_idx
            ), "'where' and 'out' should be passed as keyword arguments"

            # Convert relevant arguments to cuNumeric ndarrays
            args = tuple(
                convert_to_cunumeric_ndarray(arg)
                if idx in indices and arg is not None
                else arg
                for (idx, arg) in enumerate(args)
            )
            for k, v in kwargs.items():
                if v is None:
                    continue
                elif k == "out":
                    kwargs[k] = convert_to_cunumeric_ndarray(v, share=True)
                    if not kwargs[k].flags.writeable:
                        raise ValueError("out is not writeable")
                elif (k in keys) or (k == "where"):
                    kwargs[k] = convert_to_cunumeric_ndarray(v)

            return func(*args, **kwargs)

        return wrapper

    return decorator


def broadcast_where(
    where: Union[ndarray, None], shape: NdShape
) -> Union[ndarray, None]:
    if where is not None and where.shape != shape:
        from .._module import broadcast_to

        where = broadcast_to(where, shape)
    return where


def convert_to_cunumeric_ndarray(obj: Any, share: bool = False) -> ndarray:
    from .array import ndarray

    # If this is an instance of one of our ndarrays then we're done
    if isinstance(obj, ndarray):
        return obj
    # Ask the runtime to make a numpy thunk for this object
    thunk = runtime.get_numpy_thunk(obj, share=share)
    writeable = (
        obj.flags.writeable if isinstance(obj, np.ndarray) and share else True
    )
    return ndarray(shape=None, thunk=thunk, writeable=writeable)


def maybe_convert_to_np_ndarray(obj: Any) -> Any:
    """
    Converts cuNumeric arrays into NumPy arrays, otherwise has no effect.
    """
    from ..ma import MaskedArray
    from .array import ndarray

    if isinstance(obj, (ndarray, MaskedArray)):
        return obj.__array__()
    return obj


def check_writeable(arr: Union[ndarray, tuple[ndarray, ...], None]) -> None:
    """
    Check if the current array is writeable
    This check needs to be manually inserted
    with consideration on the behavior of the corresponding method
    """
    if arr is None:
        return
    check_list = (arr,) if not isinstance(arr, tuple) else arr
    if any(not arr.flags.writeable for arr in check_list):
        raise ValueError("array is not writeable")


def sanitize_shape(
    shape: Union[NdShapeLike, Sequence[Any], npt.NDArray[Any], ndarray]
) -> NdShape:
    from .array import ndarray

    seq: tuple[Any, ...]
    if isinstance(shape, (ndarray, np.ndarray)):
        if shape.ndim == 0:
            seq = (shape.__array__().item(),)
        else:
            seq = tuple(shape.__array__())
    elif np.isscalar(shape):
        seq = (shape,)
    else:
        seq = tuple(cast(NdShape, shape))
    try:
        # Unfortunately, we can't do this check using
        # 'isinstance(value, int)', as the values in a NumPy ndarray
        # don't satisfy the predicate (they have numpy value types,
        # such as numpy.int64).
        result = tuple(operator.index(value) for value in seq)
    except TypeError:
        raise TypeError(
            "expected a sequence of integers or a single integer, "
            f"got {shape!r}"
        )
    return result


def find_common_type(*args: ndarray) -> np.dtype[Any]:
    """Determine common type following NumPy's coercion rules.

    Parameters
    ----------
    *args : ndarray
        A list of ndarrays

    Returns
    -------
    datatype : data-type
        The type that results from applying the NumPy type promotion rules
        to the arguments.
    """
    array_types = list()
    scalars = list()
    for array in args:
        if array.ndim == 0:
            scalars.append(array.dtype.type(0))
        else:
            array_types.append(array.dtype)
    return np.result_type(*array_types, *scalars)


T = TypeVar("T")


def tuple_pop(tup: tuple[T, ...], index: int) -> tuple[T, ...]:
    return tup[:index] + tup[index + 1 :]
