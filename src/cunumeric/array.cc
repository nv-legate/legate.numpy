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

#include "cunumeric/array.h"
#include "cunumeric/runtime.h"
#include "cunumeric/random/rand_util.h"

namespace cunumeric {

Array::Array(CuNumericRuntime* runtime,
             legate::LibraryContext* context,
             std::shared_ptr<legate::LogicalStore> store)
  : runtime_(runtime), context_(context), store_(store)
{
}

int32_t Array::dim() const { return store_->dim(); }

const std::vector<size_t>& Array::shape() const { return store_->extents(); }

legate::LegateTypeCode Array::code() const { return store_->code(); }

static std::vector<int64_t> compute_strides(const std::vector<size_t>& shape)
{
  std::vector<int64_t> strides(shape.size());
  if (shape.size() > 0) {
    int64_t stride = 1;
    for (int32_t dim = shape.size() - 1; dim >= 0; --dim) {
      strides[dim] = stride;
      stride *= shape[dim];
    }
  }
  return std::move(strides);
}

void Array::random(int32_t gen_code)
{
  auto task = runtime_->create_task(CuNumericOpCode::CUNUMERIC_RAND);

  auto p_lhs = task->declare_partition(store_);

  task->add_output(store_, p_lhs);
  task->add_scalar_arg(legate::Scalar(static_cast<int32_t>(RandGenCode::UNIFORM)));
  task->add_scalar_arg(legate::Scalar(runtime_->get_next_random_epoch()));
  auto strides                    = compute_strides(shape());
  void* buffer                    = malloc(strides.size() * sizeof(int64_t) + sizeof(uint32_t));
  *static_cast<uint32_t*>(buffer) = strides.size();
  memcpy(static_cast<int8_t*>(buffer) + sizeof(uint32_t),
         strides.data(),
         strides.size() * sizeof(int64_t));
  task->add_scalar_arg(legate::Scalar(true, legate::LegateTypeCode::INT64_LT, buffer));

  runtime_->submit(std::move(task));
}

void Array::binary_op(int32_t op_code, std::shared_ptr<Array> rhs1, std::shared_ptr<Array> rhs2)
{
  auto task = runtime_->create_task(CuNumericOpCode::CUNUMERIC_BINARY_OP);

  auto p_lhs  = task->declare_partition(store_);
  auto p_rhs1 = task->declare_partition(rhs1->store_);
  auto p_rhs2 = task->declare_partition(rhs2->store_);

  task->add_output(store_, p_lhs);
  task->add_input(rhs1->store_, p_rhs1);
  task->add_input(rhs2->store_, p_rhs2);
  task->add_scalar_arg(legate::Scalar(op_code));

  task->add_constraint(align(p_lhs, p_rhs1));
  task->add_constraint(align(p_rhs1, p_rhs2));

  runtime_->submit(std::move(task));
}

void Array::unary_op(int32_t op_code, std::shared_ptr<Array> input)
{
  auto task = runtime_->create_task(CuNumericOpCode::CUNUMERIC_UNARY_OP);

  auto p_out = task->declare_partition(store_);
  auto p_in  = task->declare_partition(input->store_);

  task->add_output(store_, p_out);
  task->add_input(input->store_, p_in);
  task->add_scalar_arg(legate::Scalar(op_code));

  task->add_constraint(align(p_out, p_in));

  runtime_->submit(std::move(task));
}

}  // namespace cunumeric
