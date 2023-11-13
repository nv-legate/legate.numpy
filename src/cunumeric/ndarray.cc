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

#include "cunumeric/binary/binary_op_util.h"
#include "cunumeric/operators.h"
#include "cunumeric/random/rand_util.h"
#include "cunumeric/runtime.h"
#include "cunumeric/unary/convert_util.h"
#include "cunumeric/unary/unary_op_util.h"
#include "cunumeric/unary/unary_red_util.h"

namespace cunumeric {

namespace {

struct generate_zero_fn {
  template <legate::Type::Code CODE>
  legate::Scalar operator()()
  {
    using VAL = legate::legate_type_of<CODE>;
    return legate::Scalar(VAL(0));
  }
};

}  // namespace

NDArray::NDArray(legate::LogicalStore&& store) : store_(std::forward<legate::LogicalStore>(store))
{
}

int32_t NDArray::dim() const { return store_.dim(); }

const std::vector<size_t>& NDArray::shape() const { return store_.extents().data(); }

size_t NDArray::size() const { return store_.volume(); }

legate::Type NDArray::type() const { return store_.type(); }

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
  return strides;
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

NDArray::operator bool() const { return ((NDArray*)this)->get_read_accessor<bool, 1>()[0]; }

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
  if (size() == 0) return;

  auto runtime = CuNumericRuntime::get_runtime();

  auto task = runtime->create_task(CuNumericOpCode::CUNUMERIC_RAND);

  task.add_output(store_);
  task.add_scalar_arg(legate::Scalar(static_cast<int32_t>(RandGenCode::UNIFORM)));
  task.add_scalar_arg(legate::Scalar(runtime->get_next_random_epoch()));
  auto strides = compute_strides(shape());
  task.add_scalar_arg(legate::Scalar(strides));

  runtime->submit(std::move(task));
}

void NDArray::fill(const Scalar& value, bool argval)
{
  if (size() == 0) return;

  auto runtime = CuNumericRuntime::get_runtime();

  auto fill_value = runtime->create_scalar_store(value);

  auto task = runtime->create_task(CuNumericOpCode::CUNUMERIC_FILL);

  task.add_output(store_);
  task.add_input(fill_value);
  task.add_scalar_arg(legate::Scalar(argval));

  runtime->submit(std::move(task));
}

void NDArray::eye(int32_t k)
{
  if (size() == 0) return;

  assert(dim() == 2);

  auto zero = legate::type_dispatch(type().code(), generate_zero_fn{});
  fill(zero, false);

  auto runtime = CuNumericRuntime::get_runtime();

  auto task = runtime->create_task(CuNumericOpCode::CUNUMERIC_EYE);

  task.add_input(store_);
  task.add_output(store_);
  task.add_scalar_arg(legate::Scalar(k));

  runtime->submit(std::move(task));
}

void NDArray::bincount(NDArray rhs, std::optional<NDArray> weights /*=std::nullopt*/)
{
  if (size() == 0) return;

  assert(dim() == 1);

  auto runtime = CuNumericRuntime::get_runtime();

  if (weights.has_value()) { assert(rhs.shape() == weights.value().shape()); }

  auto zero = legate::type_dispatch(type().code(), generate_zero_fn{});
  fill(zero, false);

  auto task                     = runtime->create_task(CuNumericOpCode::CUNUMERIC_BINCOUNT);
  legate::ReductionOpKind redop = legate::ReductionOpKind::ADD;

  auto p_lhs = task.add_reduction(store_, redop);
  auto p_rhs = task.add_input(rhs.store_);
  task.add_constraint(legate::broadcast(p_lhs, {0}));
  if (weights.has_value()) {
    auto p_weight = task.add_input(weights.value().store_);
    task.add_constraint(legate::align(p_rhs, p_weight));
  }

  runtime->submit(std::move(task));
}

void NDArray::sort_task(NDArray rhs, bool argsort, bool stable)
{
  auto runtime = CuNumericRuntime::get_runtime();
  auto task    = runtime->create_task(CuNumericOpCode::CUNUMERIC_SORT);
  auto p_rhs   = task.add_input(rhs.store_);

  auto machine             = legate::Runtime::get_runtime()->get_machine();
  bool uses_unbound_output = machine.count() > 1 and rhs.dim() == 1;
  std::optional<NDArray> unbound;
  if (uses_unbound_output) {
    unbound = runtime->create_array(type());
    task.add_output(unbound.value().get_store());
  } else {
    auto p_lhs = task.add_output(store_);
    task.add_constraint(align(p_lhs, p_rhs));
  }

  if (machine.count(legate::mapping::TaskTarget::GPU) > 0)
    task.add_communicator("nccl");
  else
    task.add_communicator("cpu");

  task.add_scalar_arg(legate::Scalar(argsort));
  task.add_scalar_arg(legate::Scalar(rhs.shape()));
  task.add_scalar_arg(legate::Scalar(stable));
  runtime->submit(std::move(task));
  if (uses_unbound_output) store_ = unbound.value().get_store();
}

void NDArray::sort_swapped(NDArray rhs, bool argsort, int32_t sort_axis, bool stable)
{
  sort_axis = normalize_axis_index(sort_axis, rhs.dim());

  auto swapped      = rhs.swapaxes(sort_axis, rhs.dim() - 1);
  auto runtime      = CuNumericRuntime::get_runtime();
  auto swapped_copy = runtime->create_array(swapped.shape(), swapped.type());
  swapped_copy.assign(swapped);

  if (argsort) {
    auto sort_result = runtime->create_array(swapped_copy.shape(), type());
    sort_result.sort(swapped_copy, argsort, -1, stable);
    store_ = sort_result.swapaxes(rhs.dim() - 1, sort_axis).get_store();
  } else {
    swapped_copy.sort(swapped_copy, argsort, -1, stable);
    store_ = swapped_copy.swapaxes(rhs.dim() - 1, sort_axis).get_store();
  }
}

void NDArray::sort(NDArray rhs, bool argsort, std::optional<int32_t> axis, bool stable)
{
  if (!axis.has_value() && rhs.dim() > 1) {
    // TODO: need to flatten the rhs and sort it, the implementation depends on reshape method.
    assert(false);
  }
  int32_t computed_axis = 0;
  if (axis.has_value()) computed_axis = normalize_axis_index(axis.value(), rhs.dim());

  if (computed_axis == rhs.dim() - 1) {
    sort_task(rhs, argsort, stable);
  } else {
    sort_swapped(rhs, argsort, computed_axis, stable);
  }
}

void NDArray::sort(NDArray rhs,
                   bool argsort /*=false*/,
                   std::optional<int32_t> axis /*=-1*/,
                   std::string kind /*="quicksort"*/)
{
  if (axis.has_value() && (axis >= rhs.dim() || axis < -rhs.dim()))
    throw std::invalid_argument("invalid axis");

  if (!(kind == "quicksort" || kind == "mergesort" || kind == "heapsort" || kind == "stable"))
    throw std::invalid_argument("invalid kind");

  bool stable = (kind == "stable");
  sort(rhs, argsort, axis, stable);
}

void NDArray::trilu(NDArray rhs, int32_t k, bool lower)
{
  if (size() == 0) return;

  auto runtime = CuNumericRuntime::get_runtime();

  auto task = runtime->create_task(CuNumericOpCode::CUNUMERIC_TRILU);

  auto& out_shape = shape();
  rhs             = rhs.broadcast(out_shape, rhs.store_);

  task.add_scalar_arg(legate::Scalar(lower));
  task.add_scalar_arg(legate::Scalar(k));

  auto p_lhs = task.add_output(store_);
  auto p_rhs = task.add_input(rhs.store_);

  task.add_constraint(align(p_lhs, p_rhs));

  runtime->submit(std::move(task));
}

void NDArray::binary_op(int32_t op_code, NDArray rhs1, NDArray rhs2)
{
  if (rhs1.type() != rhs2.type()) throw std::invalid_argument("Operands must have the same type");

  if (size() == 0) return;

  auto runtime = CuNumericRuntime::get_runtime();

  auto task = runtime->create_task(CuNumericOpCode::CUNUMERIC_BINARY_OP);

  auto& out_shape = shape();
  auto rhs1_store = broadcast(out_shape, rhs1.store_);
  auto rhs2_store = broadcast(out_shape, rhs2.store_);

  auto p_lhs  = task.add_output(store_);
  auto p_rhs1 = task.add_input(rhs1_store);
  auto p_rhs2 = task.add_input(rhs2_store);
  task.add_scalar_arg(legate::Scalar(op_code));

  task.add_constraint(align(p_lhs, p_rhs1));
  task.add_constraint(align(p_rhs1, p_rhs2));

  runtime->submit(std::move(task));
}

void NDArray::binary_reduction(int32_t op_code, NDArray rhs1, NDArray rhs2)
{
  if (size() == 0) return;

  auto runtime = CuNumericRuntime::get_runtime();

  auto rhs1_store = broadcast(rhs1, rhs2);
  auto rhs2_store = broadcast(rhs2, rhs1);

  legate::ReductionOpKind redop;
  if (op_code == static_cast<int32_t>(BinaryOpCode::NOT_EQUAL)) {
    redop = runtime->get_reduction_op(UnaryRedCode::SUM);
    fill(legate::Scalar(false), false);
  } else {
    redop = runtime->get_reduction_op(UnaryRedCode::PROD);
    fill(legate::Scalar(true), false);
  }
  auto task = runtime->create_task(CuNumericOpCode::CUNUMERIC_BINARY_RED);

  task.add_reduction(store_, redop);
  auto p_rhs1 = task.add_input(rhs1_store);
  auto p_rhs2 = task.add_input(rhs2_store);
  task.add_scalar_arg(legate::Scalar(op_code));

  task.add_constraint(align(p_rhs1, p_rhs2));

  runtime->submit(std::move(task));
}

void NDArray::unary_op(int32_t op_code, NDArray input)
{
  if (size() == 0) return;

  auto runtime = CuNumericRuntime::get_runtime();

  auto task = runtime->create_task(CuNumericOpCode::CUNUMERIC_UNARY_OP);

  auto rhs = broadcast(shape(), input.store_);

  auto p_out = task.add_output(store_);
  auto p_in  = task.add_input(rhs);
  task.add_scalar_arg(legate::Scalar(op_code));

  task.add_constraint(align(p_out, p_in));

  runtime->submit(std::move(task));
}

void NDArray::unary_reduction(int32_t op_code_, NDArray input)
{
  if (size() == 0) return;

  auto runtime = CuNumericRuntime::get_runtime();

  auto op_code = static_cast<UnaryRedCode>(op_code_);

  auto identity = runtime->get_reduction_identity(op_code, type());
  fill(identity, false);

  auto task = runtime->create_task(CuNumericOpCode::CUNUMERIC_SCALAR_UNARY_RED);

  auto redop = runtime->get_reduction_op(op_code);

  task.add_reduction(store_, redop);
  task.add_input(input.store_);
  task.add_scalar_arg(legate::Scalar(op_code_));
  task.add_scalar_arg(legate::Scalar(input.shape()));

  runtime->submit(std::move(task));
}

void NDArray::dot(NDArray rhs1, NDArray rhs2)
{
  if (size() == 0) return;

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

  auto redop = runtime->get_reduction_op(UnaryRedCode::SUM);

  auto p_lhs  = task.add_reduction(lhs_s, redop);
  auto p_rhs1 = task.add_input(rhs1_s);
  auto p_rhs2 = task.add_input(rhs2_s);

  task.add_constraint(align(p_lhs, p_rhs1));
  task.add_constraint(align(p_rhs1, p_rhs2));

  runtime->submit(std::move(task));
}

void NDArray::arange(double start, double stop, double step)
{
  if (size() == 0) return;

  auto runtime = CuNumericRuntime::get_runtime();

  assert(dim() == 1);

  // TODO: Optimization when value is a scalar

  auto task = runtime->create_task(CuNumericOpCode::CUNUMERIC_ARANGE);

  task.add_output(store_);

  task.add_scalar_arg(Scalar(start));
  task.add_scalar_arg(Scalar(step));

  runtime->submit(std::move(task));
}

std::vector<NDArray> NDArray::nonzero()
{
  auto runtime = CuNumericRuntime::get_runtime();

  std::vector<NDArray> outputs;
  auto ndim = dim();
  for (int32_t i = 0; i < ndim; ++i) outputs.emplace_back(runtime->create_array(legate::int64()));

  auto task = runtime->create_task(CuNumericOpCode::CUNUMERIC_NONZERO);

  for (auto& output : outputs) { task.add_output(output.store_); }
  auto p_rhs = task.add_input(store_);

  task.add_constraint(legate::broadcast(p_rhs, legate::from_range<int32_t>(1, ndim)));

  runtime->submit(std::move(task));

  return outputs;
}

NDArray NDArray::unique()
{
  auto machine  = legate::Runtime::get_runtime()->get_machine();
  bool has_gpus = machine.count(legate::mapping::TaskTarget::GPU) > 0;

  auto runtime = CuNumericRuntime::get_runtime();
  auto result  = runtime->create_array(type());

  auto task     = runtime->create_task(CuNumericOpCode::CUNUMERIC_UNIQUE);
  auto part_out = task.declare_partition();
  auto part_in  = task.declare_partition();
  task.add_output(result.store_, part_out);
  task.add_input(store_, part_in);
  task.add_communicator("nccl");
  if (!has_gpus)
    task.add_constraint(legate::broadcast(part_in, legate::from_range<int32_t>(0, dim())));
  runtime->submit(std::move(task));
  return result;
}

NDArray NDArray::swapaxes(int32_t axis1, int32_t axis2)
{
  axis1 = normalize_axis_index(axis1, dim());
  axis2 = normalize_axis_index(axis2, dim());

  if (shape().size() == 1 || axis1 == axis2) return *this;

  auto ndim = dim();
  std::vector<int32_t> dims;
  for (auto i = 0; i < ndim; ++i) dims.push_back(i);

  if (axis1 < 0 || axis2 < 0) throw std::out_of_range("Index is out of range");

  std::swap(dims[axis1], dims[axis2]);

  auto transposed = store_.transpose(std::move(dims));
  auto runtime    = CuNumericRuntime::get_runtime();
  return runtime->create_array(std::move(transposed));
}

NDArray NDArray::as_type(const legate::Type& type)
{
  auto runtime = CuNumericRuntime::get_runtime();

  // TODO: Check if conversion is valid

  auto out = runtime->create_array(shape(), type);

  if (size() == 0) return out;

  assert(store_.type() != out.store_.type());

  auto task = runtime->create_task(CuNumericOpCode::CUNUMERIC_CONVERT);

  auto p_lhs = task.add_output(out.store_);
  auto p_rhs = task.add_input(store_);
  task.add_scalar_arg(legate::Scalar((int32_t)ConvertCode::NOOP));

  task.add_constraint(align(p_lhs, p_rhs));

  runtime->submit(std::move(task));

  return out;
}

void NDArray::create_window(int32_t op_code, int64_t M, std::vector<double> args)
{
  if (size() == 0) return;

  auto runtime = CuNumericRuntime::get_runtime();

  auto task = runtime->create_task(CuNumericOpCode::CUNUMERIC_WINDOW);

  task.add_output(store_);
  task.add_scalar_arg(legate::Scalar(op_code));
  task.add_scalar_arg(legate::Scalar(M));

  for (double arg : args) { task.add_scalar_arg(legate::Scalar(arg)); }

  runtime->submit(std::move(task));
}

void NDArray::convolve(NDArray input, NDArray filter)
{
  auto runtime = CuNumericRuntime::get_runtime();

  auto task = runtime->create_task(CuNumericOpCode::CUNUMERIC_CONVOLVE);

  auto p_filter = task.add_input(filter.store_);
  auto p_input  = task.add_input(input.store_);
  auto p_halo   = task.declare_partition();
  task.add_input(input.store_, p_halo);
  auto p_output = task.add_output(store_);
  task.add_scalar_arg(legate::Scalar(shape()));

  auto offsets = (filter.store_.extents() + size_t{1}) / size_t{2};

  task.add_constraint(legate::align(p_input, p_output));
  task.add_constraint(legate::bloat(p_input, p_halo, offsets, offsets));
  task.add_constraint(legate::broadcast(p_filter, legate::from_range<int32_t>(dim())));

  runtime->submit(std::move(task));
}

NDArray NDArray::transpose()
{
  if (dim() == 1) return NDArray(std::move(store_));
  std::vector<int32_t> axes;
  for (int32_t i = dim() - 1; i > -1; --i) axes.push_back(i);
  return transpose(axes);
}

NDArray NDArray::transpose(std::vector<int32_t> axes)
{
  if (dim() == 1) return NDArray(std::move(store_));
  if (static_cast<int32_t>(axes.size()) != dim()) {
    throw std::invalid_argument("axes must be the same size as ndim for transpose");
  }
  return NDArray(store_.transpose(std::move(axes)));
}

legate::LogicalStore NDArray::get_store() { return store_; }

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
  assert(static_cast<size_t>(result.dim()) == shape.size());
#endif

  return result;
}

legate::LogicalStore NDArray::broadcast(NDArray rhs1, NDArray rhs2)
{
  if (rhs1.shape() == rhs2.shape()) { return rhs1.store_; }
  auto out_shape = broadcast_shapes({rhs1, rhs2});
  return broadcast(out_shape, rhs1.store_);
}

/*static*/ legate::Library NDArray::get_library()
{
  return CuNumericRuntime::get_runtime()->get_library();
}

}  // namespace cunumeric
