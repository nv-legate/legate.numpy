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

#include "numpy/array.h"
#include "numpy/runtime.h"
#include "numpy/random/rand_util.h"

namespace legate {
namespace numpy {

Array::Array(NumPyRuntime* runtime,
             LibraryContext* context,
             std::vector<int64_t> shape,
             std::shared_ptr<LogicalStore> store)
  : runtime_(runtime), context_(context), shape_(std::move(shape)), store_(store)
{
}

int32_t Array::dim() const { return static_cast<int32_t>(shape_.size()); }

const std::vector<int64_t>& Array::shape() const { return shape_; }

LegateTypeCode Array::code() const { return store_->code(); }

static std::vector<int64_t> compute_strides(const std::vector<int64_t>& shape)
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
  auto task = runtime_->create_task(NumPyOpCode::NUMPY_RAND);

  task->add_output(store_);
  task->add_scalar_arg(Scalar(static_cast<int32_t>(RandGenCode::UNIFORM)));
  task->add_scalar_arg(Scalar(runtime_->get_next_random_epoch()));
  auto strides                    = compute_strides(shape_);
  void* buffer                    = malloc(strides.size() * sizeof(int64_t) + sizeof(uint32_t));
  *static_cast<uint32_t*>(buffer) = strides.size();
  memcpy(static_cast<int8_t*>(buffer) + sizeof(uint32_t),
         strides.data(),
         strides.size() * sizeof(int64_t));
  task->add_scalar_arg(Scalar(true, LegateTypeCode::INT64_LT, buffer));

  runtime_->submit(std::move(task));
}

void Array::binary_op(int32_t op_code, std::shared_ptr<Array> rhs1, std::shared_ptr<Array> rhs2)
{
  auto task = runtime_->create_task(NumPyOpCode::NUMPY_BINARY_OP);

  task->add_output(store_);
  task->add_input(rhs1->store_);
  task->add_input(rhs2->store_);
  task->add_scalar_arg(Scalar(op_code));

  runtime_->submit(std::move(task));
}

void Array::unary_op(int32_t op_code, std::shared_ptr<Array> input)
{
  auto task = runtime_->create_task(NumPyOpCode::NUMPY_UNARY_OP);

  task->add_output(store_);
  task->add_input(input->store_);
  task->add_scalar_arg(Scalar(op_code));

  runtime_->submit(std::move(task));
}

}  // namespace numpy
}  // namespace legate
