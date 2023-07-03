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

#include "cunumeric/arg_redop_register.h"

extern "C" {

void cunumeric_register_reduction_op(uintptr_t raw_type)
{
  const auto* type = reinterpret_cast<const legate::StructType*>(raw_type);
  legate::type_dispatch(type->field_type(1).code, cunumeric::register_reduction_op_fn{}, type);
}
}
