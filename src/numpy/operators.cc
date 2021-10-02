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
#include "numpy/binary/binary_op_util.h"
#include "numpy/unary/unary_op_util.h"
#include "numpy/random/rand_util.h"

namespace legate {
namespace numpy {

using ArrayP = std::shared_ptr<Array>;

ArrayP array(std::vector<size_t> shape, LegateTypeCode type)
{
  return NumPyRuntime::get_runtime()->create_array(std::move(shape), type);
}

ArrayP unary_op(UnaryOpCode op_code, ArrayP input)
{
  auto runtime = NumPyRuntime::get_runtime();
  auto out     = runtime->create_array(input->shape(), input->code());
  out->unary_op(static_cast<int32_t>(op_code), std::move(input));
  return std::move(out);
}

ArrayP binary_op(BinaryOpCode op_code, ArrayP rhs1, ArrayP rhs2)
{
  assert(rhs1->shape() == rhs2->shape());
  assert(rhs1->code() == rhs2->code());

  auto runtime = NumPyRuntime::get_runtime();
  auto out     = runtime->create_array(rhs1->shape(), rhs1->code());
  out->binary_op(static_cast<int32_t>(op_code), std::move(rhs1), std::move(rhs2));
  return std::move(out);
}

ArrayP abs(ArrayP input) { return unary_op(UnaryOpCode::ABSOLUTE, std::move(input)); }

ArrayP add(ArrayP rhs1, ArrayP rhs2)
{
  return binary_op(BinaryOpCode::ADD, std::move(rhs1), std::move(rhs2));
}

ArrayP negative(ArrayP input) { return unary_op(UnaryOpCode::NEGATIVE, std::move(input)); }

ArrayP random(std::vector<size_t> shape)
{
  auto runtime = NumPyRuntime::get_runtime();
  auto out     = runtime->create_array(std::move(shape), LegateTypeCode::DOUBLE_LT);
  out->random(static_cast<int32_t>(RandGenCode::UNIFORM));
  return out;
}

}  // namespace numpy
}  // namespace legate
