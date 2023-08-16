/* Copyright 2021-2022 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#pragma once

#include "cunumeric/cunumeric_task.h"

namespace cunumeric {

struct ContractArgs {
  legate::Store lhs;
  legate::Store rhs1;
  legate::Store rhs2;
  legate::Span<const bool> lhs_dim_mask;
  legate::Span<const bool> rhs1_dim_mask;
  legate::Span<const bool> rhs2_dim_mask;
};

class ContractTask : public CuNumericTask<ContractTask> {
 public:
  static const int TASK_ID = CUNUMERIC_CONTRACT;

 public:
  static void cpu_variant(legate::TaskContext context);
#ifdef LEGATE_USE_OPENMP
  static void omp_variant(legate::TaskContext context);
#endif
#ifdef LEGATE_USE_CUDA
  static void gpu_variant(legate::TaskContext context);
#endif
};

}  // namespace cunumeric
