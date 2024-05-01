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

from typing import Tuple

from llvmlite.ir._utils import (
    _HasMetadata,
    _StrCaching,
    _StringReferenceCaching,
)
from llvmlite.ir.module import Module
from llvmlite.ir.types import FunctionType

class Value: ...
class NamedValue(_StrCaching, _StringReferenceCaching, Value): ...
class Block(NamedValue): ...
class _BaseArgument(NamedValue): ...
class Argument(_BaseArgument): ...
class _ConstOpMixin: ...
class GlobalValue(NamedValue, _ConstOpMixin, _HasMetadata): ...

class Function(GlobalValue):
    args: Tuple[Argument]

    def __init__(self, module: Module, ftype: FunctionType, name: str): ...
    def append_basic_block(self, name: str) -> Block: ...
