#!/usr/bin/env python

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

import sys

import legate.tester
from legate.tester import CustomTest
from legate.tester.config import Config
from legate.tester.test_plan import TestPlan
from legate.tester.test_system import TestSystem

legate.tester.CUSTOM_FILES = [
    CustomTest("examples/quantiles.py"),
    CustomTest("examples/sort.py"),
    CustomTest("tests/integration/test_argsort.py"),
    CustomTest("tests/integration/test_msort.py"),
    CustomTest("tests/integration/test_nanpercentiles.py"),
    CustomTest("tests/integration/test_nanquantiles.py"),
    CustomTest("tests/integration/test_partition.py"),
    CustomTest("tests/integration/test_percentiles.py"),
    CustomTest("tests/integration/test_quantiles.py"),
    CustomTest("tests/integration/test_sort_complex.py"),
    CustomTest("tests/integration/test_sort.py"),
    CustomTest("tests/integration/test_unique.py"),
]

if __name__ == "__main__":
    config = Config(sys.argv)

    system = TestSystem(dry_run=config.dry_run)

    plan = TestPlan(config, system)

    sys.exit(plan.execute())
