/* Copyright 2023 NVIDIA Corporation
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

#include "legate.h"
#include "cunumeric/cunumeric_c.h"
#include "cunumeric/arg.h"

namespace cunumeric {

struct register_reduction_op_fn {
  template <legate::Type::Code CODE, std::enable_if_t<!legate::is_complex<CODE>::value>* = nullptr>
  ReductionOpIds operator()()
  {
    using VAL = legate::legate_type_of<CODE>;
    ReductionOpIds result;
    auto runtime = legate::Runtime::get_runtime();
    auto context = runtime->find_library("cunumeric");
    result.argmax_redop_id =
      context.register_reduction_operator<ArgmaxReduction<VAL>>(next_reduction_operator_id());
    result.argmin_redop_id =
      context.register_reduction_operator<ArgminReduction<VAL>>(next_reduction_operator_id());
    return result;
  }

  template <legate::Type::Code CODE, std::enable_if_t<legate::is_complex<CODE>::value>* = nullptr>
  ReductionOpIds operator()()
  {
    LEGATE_ABORT;
    return ReductionOpIds{};
  }

  static int32_t next_reduction_operator_id();
};

}  // namespace cunumeric
