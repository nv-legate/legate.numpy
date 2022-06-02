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

#include "cunumeric/operators.h"
#include "cunumeric/array.h"
#include "cunumeric/runtime.h"
#include "cunumeric/binary/binary_op_util.h"
#include "cunumeric/unary/unary_op_util.h"
#include "cunumeric/random/rand_util.h"

namespace cunumeric {

using ArrayP = std::shared_ptr<Array>;

ArrayP array(std::vector<size_t> shape, legate::LegateTypeCode type)
{
  return CuNumericRuntime::get_runtime()->create_array(std::move(shape), type);
}

ArrayP unary_op(UnaryOpCode op_code, ArrayP input)
{
  auto runtime = CuNumericRuntime::get_runtime();
  auto out     = runtime->create_array(input->shape(), input->code());
  out->unary_op(static_cast<int32_t>(op_code), std::move(input));
  return std::move(out);
}

ArrayP unary_reduction(UnaryRedCode op_code, ArrayP input)
{
  auto runtime = CuNumericRuntime::get_runtime();
  auto out     = runtime->create_array({1}, input->code());
  out->unary_reduction(static_cast<int32_t>(op_code), std::move(input));
  return std::move(out);
}

ArrayP binary_op(BinaryOpCode op_code, ArrayP rhs1, ArrayP rhs2)
{
  assert(rhs1->shape() == rhs2->shape());
  assert(rhs1->code() == rhs2->code());

  auto runtime = CuNumericRuntime::get_runtime();
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
  auto runtime = CuNumericRuntime::get_runtime();
  auto out     = runtime->create_array(std::move(shape), legate::LegateTypeCode::DOUBLE_LT);
  out->random(static_cast<int32_t>(RandGenCode::UNIFORM));
  return std::move(out);
}

ArrayP full(std::vector<size_t> shape, const Scalar& value)
{
  auto runtime = CuNumericRuntime::get_runtime();
  auto out     = runtime->create_array(std::move(shape), value.code());
  out->fill(value, false);
  return std::move(out);
}

ArrayP dot(ArrayP rhs1, ArrayP rhs2)
{
  if (rhs1->dim() != 2 || rhs2->dim() != 2) {
    fprintf(stderr, "cunumeric::dot only supports matrices now");
    LEGATE_ABORT;
  }

  auto& rhs1_shape = rhs1->shape();
  auto& rhs2_shape = rhs2->shape();

  if (rhs1_shape[1] != rhs2_shape[0]) {
    fprintf(stderr,
            "Incompatible matrices: (%zd, %zd) x (%zd, %zd)\n",
            rhs1_shape[0],
            rhs1_shape[1],
            rhs2_shape[0],
            rhs2_shape[1]);
    LEGATE_ABORT;
  }

  auto runtime = CuNumericRuntime::get_runtime();
  std::vector<size_t> shape;
  shape.push_back(rhs1_shape[0]);
  shape.push_back(rhs2_shape[1]);

  auto out = runtime->create_array(std::move(shape), rhs1->code());
  out->dot(std::move(rhs1), std::move(rhs2));
  return std::move(out);
}

ArrayP sum(ArrayP input) { return unary_reduction(UnaryRedCode::SUM, std::move(input)); }

}  // namespace cunumeric
