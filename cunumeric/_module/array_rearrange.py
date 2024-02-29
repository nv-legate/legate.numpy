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

from typing import TYPE_CHECKING, Optional

from .._array.util import add_boilerplate

if TYPE_CHECKING:
    from .._array.array import ndarray
    from ..types import NdShapeLike


@add_boilerplate("m")
def flip(m: ndarray, axis: Optional[NdShapeLike] = None) -> ndarray:
    """
    Reverse the order of elements in an array along the given axis.

    The shape of the array is preserved, but the elements are reordered.

    Parameters
    ----------
    m : array_like
        Input array.
    axis : None or int or tuple[int], optional
         Axis or axes along which to flip over. The default, axis=None, will
         flip over all of the axes of the input array.  If axis is negative it
         counts from the last to the first axis.

         If axis is a tuple of ints, flipping is performed on all of the axes
         specified in the tuple.

    Returns
    -------
    out : array_like
        A new array that is constructed from `m` with the entries of axis
        reversed.

    See Also
    --------
    numpy.flip

    Availability
    --------
    Single GPU, Single CPU

    Notes
    -----
    cuNumeric implementation doesn't return a view, it returns a new array
    """
    return m.flip(axis=axis)


@add_boilerplate("m")
def flipud(m: ndarray) -> ndarray:
    """
    Reverse the order of elements along axis 0 (up/down).

    For a 2-D array, this flips the entries in each column in the up/down
    direction. Rows are preserved, but appear in a different order than before.

    Parameters
    ----------
    m : array_like
        Input array.

    Returns
    -------
    out : array_like
        A new array that is constructed from `m` with rows reversed.

    See Also
    --------
    numpy.flipud

    Availability
    --------
    Single GPU, Single CPU

    Notes
    -----
    cuNumeric implementation doesn't return a view, it returns a new array
    """
    if m.ndim < 1:
        raise ValueError("Input must be >= 1-d.")
    return flip(m, axis=0)


@add_boilerplate("m")
def fliplr(m: ndarray) -> ndarray:
    """
    Reverse the order of elements along axis 1 (left/right).

    For a 2-D array, this flips the entries in each row in the left/right
    direction. Columns are preserved, but appear in a different order than
    before.

    Parameters
    ----------
    m : array_like
        Input array, must be at least 2-D.

    Returns
    -------
    f : ndarray
        A new array that is constructed from `m` with the columns reversed.

    See Also
    --------
    numpy.fliplr

    Availability
    --------
    Single GPU, Single CPU

    Notes
    -----
    cuNumeric implementation doesn't return a view, it returns a new array
    """
    if m.ndim < 2:
        raise ValueError("Input must be >= 2-d.")
    return flip(m, axis=1)
