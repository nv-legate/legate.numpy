/* Copyright 2021 NVIDIA Corporation
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

#include "numpy/operators.h"
#include "numpy/array.h"
#include "numpy/runtime.h"
#include "numpy/random/rand_util.h"

namespace legate {
namespace numpy {

std::shared_ptr<Array> array(std::vector<int64_t> shape, LegateTypeCode type)
{
  return NumPyRuntime::get_runtime()->create_array(std::move(shape), type);
}

std::shared_ptr<Array> random(std::vector<int64_t> shape)
{
  auto runtime = NumPyRuntime::get_runtime();
  auto out     = runtime->create_array(std::move(shape), LegateTypeCode::DOUBLE_LT);
  out->random(static_cast<int32_t>(RandGenCode::UNIFORM));
  return out;
}

}  // namespace numpy
}  // namespace legate
