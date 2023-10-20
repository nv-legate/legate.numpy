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
from __future__ import annotations

from typing import TYPE_CHECKING, cast

from legate.core import broadcast, get_legate_runtime

from cunumeric.config import CuNumericOpCode

from .cholesky import transpose_copy_single

# from .exception import LinAlgError

if TYPE_CHECKING:
    from legate.core import Library, LogicalStore

    from ..deferred import DeferredArray


def solve_single(library: Library, a: LogicalStore, b: LogicalStore) -> None:
    task = get_legate_runtime().create_auto_task(
        library, CuNumericOpCode.SOLVE
    )
    # TODO: We need to put back the precise Python exception support
    # task.throws_exception(LinAlgError)
    task.throws_exception(True)
    p_a = task.add_input(a)
    p_b = task.add_input(b)
    task.add_output(a, p_a)
    task.add_output(b, p_b)

    task.add_constraint(broadcast(p_a))
    task.add_constraint(broadcast(p_b))

    task.execute()


def solve(output: DeferredArray, a: DeferredArray, b: DeferredArray) -> None:
    from ..deferred import DeferredArray

    runtime = output.runtime
    library = output.library

    a_copy = cast(
        DeferredArray,
        runtime.create_empty_thunk(a.shape, dtype=a.base.type, inputs=(a,)),
    )
    transpose_copy_single(library, a.base, a_copy.base)

    if b.ndim > 1:
        transpose_copy_single(library, b.base, output.base)
    else:
        output.copy(b)

    solve_single(library, a_copy.base, output.base)
