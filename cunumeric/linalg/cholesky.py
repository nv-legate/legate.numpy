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

from legate.core import broadcast, get_legate_runtime, types as ty

from cunumeric.config import CuNumericOpCode

# from legate.core.shape import Shape
# from legate.settings import settings


# from .exception import LinAlgError

legate_runtime = get_legate_runtime()

if TYPE_CHECKING:
    from legate.core import Library, LogicalStore, LogicalStorePartition

    from ..deferred import DeferredArray

    # from ..runtime import Runtime


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
    launch_domain: list[int],
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
    # TODO: We need to put back the precise Python exception support
    # task.throws_exception(LinAlgError)
    task.throws_exception(True)
    task.add_output(output)
    task.add_input(output)
    task.execute()


# def potrf(library: Library, p_output: LogicalStorePartition, i: int) -> None:
#    launch_domain = [1, 1]
#    task = legate_runtime.create_manual_task(
#        library, CuNumericOpCode.POTRF, launch_domain
#    )
#    # TODO: We need to put back the precise Python exception support
#    # task.throws_exception(LinAlgError)
#    task.throws_exception(True)
#    task.add_output(p_output)
#    task.add_input(p_output)
#    task.execute()


# def trsm(
#   library: Library, p_output: LogicalStorePartition, i: int, lo: int, hi: int
# ) -> None:
#    if lo >= hi:
#        return
#
#    rhs = p_output.get_child_store(i, i)
#    lhs = p_output
#
#    launch_domain = [hi - lo, 1]
#    task = legate_runtime.create_manual_task(
#        library, CuNumericOpCode.TRSM, launch_domain
#    )
#    task.add_output(lhs)
#    task.add_input(rhs)
#    task.add_input(lhs)
#    task.execute()


# def syrk(
#    library: Library, p_output: LogicalStorePartition, k: int, i: int
# ) -> None:
#    rhs = p_output.get_child_store(k, i)
#    lhs = p_output
#
#    launch_domain = [1, 1]
#    task = legate_runtime.create_manual_task(
#        library, CuNumericOpCode.SYRK, launch_domain
#    )
#    task.add_output(lhs)
#    task.add_input(rhs)
#    task.add_input(lhs)
#    task.execute()


# def gemm(
#    library: Library,
#    p_output: LogicalStorePartition,
#    k: int,
#    i: int,
#    lo: int,
#    hi: int,
# ) -> None:
#    if lo >= hi:
#        return
#
#    rhs2 = p_output.get_child_store(k, i)
#    lhs = p_output
#    rhs1 = p_output
#
#    launch_domain = [hi - lo, 1]
#    task = legate_runtime.create_manual_task(
#        library, CuNumericOpCode.GEMM, launch_domain
#    )
#    task.add_output(lhs)
#    task.add_input(rhs1, proj=lambda p: (p[0], i))
#    task.add_input(rhs2)
#    task.add_input(lhs)
#    task.execute()


MIN_CHOLESKY_TILE_SIZE = 2048
MIN_CHOLESKY_MATRIX_SIZE = 8192


# TODO: We need a better cost model
# def choose_color_shape(runtime: Runtime, shape: Shape) -> Shape:
#    if settings.test():
#        num_tiles = runtime.num_procs * 2
#        return Shape((num_tiles, num_tiles))
#
#    extent = shape[0]
#    # If there's only one processor or the matrix is too small,
#    # don't even bother to partition it at all
#    if runtime.num_procs == 1 or extent <= MIN_CHOLESKY_MATRIX_SIZE:
#        return Shape((1, 1))
#
#    # If the matrix is big enough to warrant partitioning,
#    # pick the granularity that the tile size is greater than a threshold
#    num_tiles = runtime.num_procs
#    max_num_tiles = runtime.num_procs * 4
#    while (
#        (extent + num_tiles - 1) // num_tiles > MIN_CHOLESKY_TILE_SIZE
#        and num_tiles * 2 <= max_num_tiles
#    ):
#        num_tiles *= 2
#
#    return Shape((num_tiles, num_tiles))


def tril_single(library: Library, output: LogicalStore) -> None:
    task = legate_runtime.create_auto_task(library, CuNumericOpCode.TRILU)
    task.add_output(output)
    task.add_input(output)
    task.add_scalar_arg(True, ty.bool_)
    task.add_scalar_arg(0, ty.int32)
    # Add a fake task argument to indicate that this is for Cholesky
    task.add_scalar_arg(True, ty.bool_)

    task.execute()


# def tril(library: Library, p_output: LogicalStorePartition, n: int) -> None:
#    launch_domain = [n, n]
#    task = legate_runtime.create_manual_task(
#        library, CuNumericOpCode.TRILU, launch_domain
#    )
#
#    task.add_output(p_output)
#    task.add_input(p_output)
#    task.add_scalar_arg(True, ty.bool_)
#    task.add_scalar_arg(0, ty.int32)
#    # Add a fake task argument to indicate that this is for Cholesky
#    task.add_scalar_arg(True, ty.bool_)
#
#    task.execute()


# TODO: Put back this parallel Cholesky implementation
# def cholesky(
#    output: DeferredArray, input: DeferredArray, no_tril: bool
# ) -> None:
#    runtime = output.runtime
#    library = output.library
#
#    if runtime.num_procs == 1:
#        transpose_copy_single(library, input.base, output.base)
#        potrf_single(library, output.base)
#        if not no_tril:
#            tril_single(library, output.base)
#        return
#
#    shape = output.base.shape
#    initial_color_shape = choose_color_shape(runtime, shape)
#    tile_shape = (shape + initial_color_shape - 1) // initial_color_shape
#    color_shape = (shape + tile_shape - 1) // tile_shape
#    n = color_shape[0]
#
#    p_input = input.base.partition_by_tiling(tile_shape)
#    p_output = output.base.partition_by_tiling(tile_shape)
#    transpose_copy(library, color_shape, p_input, p_output)
#
#    for i in range(n):
#        potrf(library, p_output, i)
#        trsm(library, p_output, i, i + 1, n)
#        for k in range(i + 1, n):
#            syrk(library, p_output, k, i)
#            gemm(library, p_output, k, i, k + 1, n)
#
#    if no_tril:
#        return
#
#    tril(library, p_output, n)


def cholesky(
    output: DeferredArray, input: DeferredArray, no_tril: bool
) -> None:
    library = output.library
    transpose_copy_single(library, input.base, output.base)
    potrf_single(library, output.base)
    if not no_tril:
        tril_single(library, output.base)
