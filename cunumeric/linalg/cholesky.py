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
from __future__ import annotations

from typing import TYPE_CHECKING

from legate.core import (
    broadcast,
    constant,
    dimension,
    get_legate_runtime,
    types as ty,
)
from legate.settings import settings

from cunumeric.config import CuNumericOpCode

from .exception import LinAlgError

legate_runtime = get_legate_runtime()

if TYPE_CHECKING:
    from legate.core import Library, LogicalStore, LogicalStorePartition

    from ..deferred import DeferredArray
    from ..runtime import Runtime


legate_runtime = get_legate_runtime()


def transpose_copy_single(
    library: Library, input: LogicalStore, output: LogicalStore
) -> None:
    task = legate_runtime.create_auto_task(
        library, CuNumericOpCode.TRANSPOSE_COPY_2D
    )
    p_out = task.add_output(output)
    p_in = task.add_input(input)
    # Output has the same shape as input, but is mapped
    # to a column major instance
    task.add_scalar_arg(False, ty.bool_)

    task.add_constraint(broadcast(p_out))
    task.add_constraint(broadcast(p_in))

    task.execute()


def transpose_copy(
    library: Library,
    launch_domain: tuple[int, ...],
    p_input: LogicalStorePartition,
    p_output: LogicalStorePartition,
) -> None:
    task = legate_runtime.create_manual_task(
        library,
        CuNumericOpCode.TRANSPOSE_COPY_2D,
        launch_domain,
    )
    task.add_output(p_output)
    task.add_input(p_input)
    # Output has the same shape as input, but is mapped
    # to a column major instance
    task.add_scalar_arg(False, ty.bool_)

    task.execute()


def potrf_single(library: Library, output: LogicalStore) -> None:
    task = legate_runtime.create_auto_task(library, CuNumericOpCode.POTRF)
    task.throws_exception(LinAlgError)
    task.add_output(output)
    task.add_input(output)
    task.execute()


def potrf(library: Library, p_output: LogicalStorePartition, i: int) -> None:
    task = legate_runtime.create_manual_task(
        library, CuNumericOpCode.POTRF, (i + 1, i + 1), lower_bounds=(i, i)
    )
    task.throws_exception(LinAlgError)
    task.add_output(p_output)
    task.add_input(p_output)
    task.execute()


def trsm(
    library: Library, p_output: LogicalStorePartition, i: int, lo: int, hi: int
) -> None:
    if lo >= hi:
        return

    rhs = p_output.get_child_store(i, i)
    lhs = p_output

    task = legate_runtime.create_manual_task(
        library, CuNumericOpCode.TRSM, (hi, i + 1), lower_bounds=(lo, i)
    )
    task.add_output(lhs)
    task.add_input(rhs)
    task.add_input(lhs)
    task.execute()


def syrk(
    library: Library, p_output: LogicalStorePartition, k: int, i: int
) -> None:
    rhs = p_output.get_child_store(k, i)
    lhs = p_output

    task = legate_runtime.create_manual_task(
        library, CuNumericOpCode.SYRK, (k + 1, k + 1), lower_bounds=(k, k)
    )
    task.add_output(lhs)
    task.add_input(rhs)
    task.add_input(lhs)
    task.execute()


def gemm(
    library: Library,
    p_output: LogicalStorePartition,
    k: int,
    i: int,
    lo: int,
    hi: int,
) -> None:
    if lo >= hi:
        return

    rhs2 = p_output.get_child_store(k, i)
    lhs = p_output
    rhs1 = p_output

    task = legate_runtime.create_manual_task(
        library, CuNumericOpCode.GEMM, (hi, k + 1), lower_bounds=(lo, k)
    )
    task.add_output(lhs)
    task.add_input(rhs1, (dimension(0), constant(i)))
    task.add_input(rhs2)
    task.add_input(lhs)
    task.execute()


MIN_CHOLESKY_TILE_SIZE = 2048
MIN_CHOLESKY_MATRIX_SIZE = 8192


# TODO: We need a better cost model
def choose_color_shape(
    runtime: Runtime, shape: tuple[int, ...]
) -> tuple[int, ...]:
    if settings.test():
        num_tiles = runtime.num_procs * 2
        return (num_tiles, num_tiles)

    extent = shape[0]
    # If there's only one processor or the matrix is too small,
    # don't even bother to partition it at all
    if runtime.num_procs == 1 or extent <= MIN_CHOLESKY_MATRIX_SIZE:
        return (1, 1)

    # If the matrix is big enough to warrant partitioning,
    # pick the granularity that the tile size is greater than a threshold
    num_tiles = runtime.num_procs
    max_num_tiles = runtime.num_procs * 4
    while (
        (extent + num_tiles - 1) // num_tiles > MIN_CHOLESKY_TILE_SIZE
        and num_tiles * 2 <= max_num_tiles
    ):
        num_tiles *= 2

    return (num_tiles, num_tiles)


def tril_single(library: Library, output: LogicalStore) -> None:
    task = legate_runtime.create_auto_task(library, CuNumericOpCode.TRILU)
    task.add_output(output)
    task.add_input(output)
    task.add_scalar_arg(True, ty.bool_)
    task.add_scalar_arg(0, ty.int32)
    # Add a fake task argument to indicate that this is for Cholesky
    task.add_scalar_arg(True, ty.bool_)

    task.execute()


def tril(library: Library, p_output: LogicalStorePartition, n: int) -> None:
    task = legate_runtime.create_manual_task(
        library, CuNumericOpCode.TRILU, (n, n)
    )

    task.add_output(p_output)
    task.add_input(p_output)
    task.add_scalar_arg(True, ty.bool_)
    task.add_scalar_arg(0, ty.int32)
    # Add a fake task argument to indicate that this is for Cholesky
    task.add_scalar_arg(True, ty.bool_)

    task.execute()


def _rounding_divide(
    lhs: tuple[int, ...], rhs: tuple[int, ...]
) -> tuple[int, ...]:
    return tuple((lh + rh - 1) // rh for (lh, rh) in zip(lhs, rhs))


def cholesky(
    output: DeferredArray, input: DeferredArray, no_tril: bool
) -> None:
    runtime = output.runtime
    library = runtime.library

    if runtime.num_procs == 1:
        transpose_copy_single(library, input.base, output.base)
        potrf_single(library, output.base)
        if not no_tril:
            tril_single(library, output.base)
        return

    shape = tuple(output.base.shape)
    initial_color_shape = choose_color_shape(runtime, shape)
    tile_shape = _rounding_divide(shape, initial_color_shape)
    color_shape = _rounding_divide(shape, tile_shape)
    n = color_shape[0]

    p_input = input.base.partition_by_tiling(tile_shape)
    p_output = output.base.partition_by_tiling(tile_shape)
    transpose_copy(library, color_shape, p_input, p_output)

    for i in range(n):
        potrf(library, p_output, i)
        trsm(library, p_output, i, i + 1, n)
        for k in range(i + 1, n):
            syrk(library, p_output, k, i)
            gemm(library, p_output, k, i, k + 1, n)

    if no_tril:
        return

    tril(library, p_output, n)
