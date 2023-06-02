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

#include "cunumeric/ndarray.h"

#include "cunumeric/operators.h"
#include "cunumeric/runtime.h"
#include "cunumeric/random/rand_util.h"
#include "cunumeric/unary/unary_op_util.h"
#include "cunumeric/unary/unary_red_util.h"

namespace cunumeric {

NDArray::NDArray(legate::LogicalStore&& store) : store_(std::forward<legate::LogicalStore>(store))
{
}

int32_t NDArray::dim() const { return store_.dim(); }

const std::vector<size_t>& NDArray::shape() const { return store_.extents().data(); }

const legate::Type& NDArray::type() const { return store_.type(); }

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

NDArray NDArray::operator+(const NDArray& other) const { return add(*this, other); }

NDArray NDArray::operator+(const legate::Scalar& other) const
{
  auto runtime = CuNumericRuntime::get_runtime();
  auto scalar  = runtime->create_scalar_store(other);
  return operator+(NDArray(std::move(scalar)));
}

NDArray& NDArray::operator+=(const NDArray& other)
{
  add(*this, other, *this);
  return *this;
}

NDArray NDArray::operator*(const NDArray& other) const { return multiply(*this, other); }

NDArray NDArray::operator*(const legate::Scalar& other) const
{
  auto runtime = CuNumericRuntime::get_runtime();
  auto scalar  = runtime->create_scalar_store(other);
  return operator*(NDArray(std::move(scalar)));
}

NDArray& NDArray::operator*=(const NDArray& other)
{
  multiply(*this, other, *this);
  return *this;
}

NDArray NDArray::operator[](std::initializer_list<slice> slices) const
{
  if (slices.size() > static_cast<size_t>(dim())) {
    std::string err_msg = "Can't slice a " + std::to_string(dim()) + "-D ndarray with " +
                          std::to_string(slices.size()) + " slices";
    throw std::invalid_argument(std::move(err_msg));
  }

  uint32_t dim = 0;
  auto sliced  = store_;
  for (const auto& sl : slices) {
    sliced = sliced.slice(0, sl);
    ++dim;
  }

  return NDArray(std::move(sliced));
}

void NDArray::assign(const NDArray& other)
{
  unary_op(static_cast<int32_t>(UnaryOpCode::COPY), other);
}

void NDArray::assign(const legate::Scalar& other)
{
  auto runtime = CuNumericRuntime::get_runtime();
  auto scalar  = runtime->create_scalar_store(other);
  assign(NDArray(std::move(scalar)));
}

void NDArray::random(int32_t gen_code)
{
  auto runtime = CuNumericRuntime::get_runtime();

  auto task = runtime->create_task(CuNumericOpCode::CUNUMERIC_RAND);

  auto p_lhs = task->declare_partition();

  task->add_output(store_, p_lhs);
  task->add_scalar_arg(legate::Scalar(static_cast<int32_t>(RandGenCode::UNIFORM)));
  task->add_scalar_arg(legate::Scalar(runtime->get_next_random_epoch()));
  auto strides = compute_strides(shape());
  task->add_scalar_arg(legate::Scalar(strides));

  runtime->submit(std::move(task));
}

void NDArray::fill(const Scalar& value, bool argval)
{
  auto runtime = CuNumericRuntime::get_runtime();

  auto fill_value = runtime->create_scalar_store(value);

  auto task         = runtime->create_task(CuNumericOpCode::CUNUMERIC_FILL);
  auto p_lhs        = task->declare_partition();
  auto p_fill_value = task->declare_partition();

  task->add_output(store_, p_lhs);
  task->add_input(fill_value, p_fill_value);
  task->add_scalar_arg(legate::Scalar(argval));

  runtime->submit(std::move(task));
}

void NDArray::binary_op(int32_t op_code, NDArray rhs1, NDArray rhs2)
{
  if (rhs1.type() != rhs2.type()) throw std::invalid_argument("Operands must have the same type");

  auto runtime = CuNumericRuntime::get_runtime();

  auto task = runtime->create_task(CuNumericOpCode::CUNUMERIC_BINARY_OP);

  auto p_lhs  = task->declare_partition();
  auto p_rhs1 = task->declare_partition();
  auto p_rhs2 = task->declare_partition();

  auto& out_shape = shape();
  auto rhs1_store = broadcast(out_shape, rhs1.store_);
  auto rhs2_store = broadcast(out_shape, rhs2.store_);

  task->add_output(store_, p_lhs);
  task->add_input(rhs1_store, p_rhs1);
  task->add_input(rhs2_store, p_rhs2);
  task->add_scalar_arg(legate::Scalar(op_code));

  task->add_constraint(align(p_lhs, p_rhs1));
  task->add_constraint(align(p_rhs1, p_rhs2));

  runtime->submit(std::move(task));
}

void NDArray::unary_op(int32_t op_code, NDArray input)
{
  auto runtime = CuNumericRuntime::get_runtime();

  auto task = runtime->create_task(CuNumericOpCode::CUNUMERIC_UNARY_OP);

  auto p_out = task->declare_partition();
  auto p_in  = task->declare_partition();

  auto rhs = broadcast(shape(), input.store_);

  task->add_output(store_, p_out);
  task->add_input(rhs, p_in);
  task->add_scalar_arg(legate::Scalar(op_code));

  task->add_constraint(align(p_out, p_in));

  runtime->submit(std::move(task));
}

void NDArray::unary_reduction(int32_t op_code_, NDArray input)
{
  auto runtime = CuNumericRuntime::get_runtime();

  auto op_code = static_cast<UnaryRedCode>(op_code_);

  auto identity = runtime->get_reduction_identity(op_code, type());
  fill(identity, false);

  auto task = runtime->create_task(CuNumericOpCode::CUNUMERIC_SCALAR_UNARY_RED);

  auto p_out = task->declare_partition();
  auto p_in  = task->declare_partition();

  auto redop = runtime->get_reduction_op(op_code, type());

  task->add_reduction(store_, redop, p_out);
  task->add_input(input.store_, p_in);
  task->add_scalar_arg(legate::Scalar(op_code_));
  task->add_scalar_arg(legate::Scalar(input.shape()));

  runtime->submit(std::move(task));
}

void NDArray::dot(NDArray rhs1, NDArray rhs2)
{
  auto runtime = CuNumericRuntime::get_runtime();

  auto identity = runtime->get_reduction_identity(UnaryRedCode::SUM, type());
  fill(identity, false);

  assert(dim() == 2 && rhs1.dim() == 2 && rhs2.dim() == 2);

  auto m = rhs1.shape()[0];
  auto n = rhs2.shape()[1];
  auto k = rhs1.shape()[1];

  auto lhs_s  = store_.promote(1, k);
  auto rhs1_s = rhs1.store_.promote(2, n);
  auto rhs2_s = rhs2.store_.promote(0, m);

  auto task = runtime->create_task(CuNumericOpCode::CUNUMERIC_MATMUL);

  auto p_lhs  = task->declare_partition();
  auto p_rhs1 = task->declare_partition();
  auto p_rhs2 = task->declare_partition();

  auto redop = runtime->get_reduction_op(UnaryRedCode::SUM, type());

  task->add_reduction(lhs_s, redop, p_lhs);
  task->add_input(rhs1_s, p_rhs1);
  task->add_input(rhs2_s, p_rhs2);

  task->add_constraint(align(p_lhs, p_rhs1));
  task->add_constraint(align(p_rhs1, p_rhs2));

  runtime->submit(std::move(task));
}

std::vector<NDArray> NDArray::nonzero()
{
  auto runtime = CuNumericRuntime::get_runtime();

  std::vector<NDArray> outputs;
  auto ndim = dim();
  for (int32_t i = 0; i < ndim; ++i) outputs.emplace_back(runtime->create_array(legate::int64()));

  auto task = runtime->create_task(CuNumericOpCode::CUNUMERIC_NONZERO);

  auto p_rhs = task->declare_partition();

  for (auto& output : outputs) {
    auto p_lhs = task->declare_partition();
    task->add_output(output.store_, p_lhs);
  }
  task->add_input(store_, p_rhs);

  task->add_constraint(legate::broadcast(p_rhs, legate::from_range<int32_t>(1, ndim)));

  runtime->submit(std::move(task));

  return std::move(outputs);
}

NDArray NDArray::unique()
{
  auto& machine = legate::Runtime::get_runtime()->get_machine();
  bool has_gpus = machine.count(legate::mapping::TaskTarget::GPU) > 0;

  auto runtime = CuNumericRuntime::get_runtime();
  auto result  = runtime->create_array(type());

  auto task     = runtime->create_task(CuNumericOpCode::CUNUMERIC_UNIQUE);
  auto part_out = task->declare_partition();
  auto part_in  = task->declare_partition();
  task->add_output(result.store_, part_out);
  task->add_input(store_, part_in);
  task->add_communicator("nccl");
  if (!has_gpus)
    task->add_constraint(legate::broadcast(part_in, legate::from_range<int32_t>(0, dim())));
  runtime->submit(std::move(task));
  return result;
}

legate::LogicalStore NDArray::broadcast(const std::vector<size_t>& shape,
                                        legate::LogicalStore& store)
{
  int32_t diff = static_cast<int32_t>(shape.size()) - store.dim();

#ifdef DEBUG_CUNUMERIC
  assert(diff >= 0);
#endif

  auto result = store;
  for (int32_t dim = 0; dim < diff; ++dim) result = result.promote(dim, shape[dim]);

  std::vector<size_t> orig_shape = result.extents().data();
  for (uint32_t dim = 0; dim < shape.size(); ++dim)
    if (orig_shape[dim] != shape[dim]) {
#ifdef DEBUG_CUNUMERIC
      assert(orig_shape[dim] == 1);
#endif
      result = result.project(dim, 0).promote(dim, shape[dim]);
    }

#ifdef DEBUG_CUNUMERIC
  assert(result.dim() == shape.size());
#endif

  return std::move(result);
}

/*static*/ legate::LibraryContext* NDArray::get_context()
{
  return CuNumericRuntime::get_runtime()->get_context();
}

}  // namespace cunumeric
