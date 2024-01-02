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

#include "cunumeric/runtime.h"

#include "cunumeric/ndarray.h"

namespace cunumeric {

/*static*/ CuNumericRuntime* CuNumericRuntime::runtime_;

static std::map<std::pair<UnaryRedCode, legate::Type::Code>, Scalar> identities;

extern void bootstrapping_callback(Legion::Machine machine,
                                   Legion::Runtime* runtime,
                                   const std::set<Legion::Processor>& local_procs);

void initialize(int32_t argc, char** argv) { cunumeric_perform_registration(); }

CuNumericRuntime::CuNumericRuntime(legate::Runtime* legate_runtime, legate::Library library)
  : legate_runtime_(legate_runtime), library_(library)
{
}

NDArray CuNumericRuntime::create_array(const legate::Type& type)
{
  auto store = legate_runtime_->create_store(type);
  return NDArray(std::move(store));
}

NDArray CuNumericRuntime::create_array(std::vector<size_t> shape,
                                       const legate::Type& type,
                                       bool optimize_scalar)
{
  auto store = legate_runtime_->create_store(legate::Shape{shape}, type, optimize_scalar);
  return NDArray(std::move(store));
}

NDArray CuNumericRuntime::create_array(legate::LogicalStore&& store)
{
  return NDArray(std::move(store));
}

legate::LogicalStore CuNumericRuntime::create_scalar_store(const Scalar& value)
{
  return legate_runtime_->create_store(value);
}

struct generate_identity_fn {
  template <UnaryRedCode OP>
  struct generator {
    template <legate::Type::Code CODE,
              std::enable_if_t<UnaryRedOp<OP, CODE>::valid && !is_arg_reduce<OP>::value>* = nullptr>
    Scalar operator()(const legate::Type&)
    {
      auto value = UnaryRedOp<OP, CODE>::OP::identity;
      return Scalar(value);
    }

    template <legate::Type::Code CODE,
              std::enable_if_t<UnaryRedOp<OP, CODE>::valid && is_arg_reduce<OP>::value>* = nullptr>
    Scalar operator()(const legate::Type& type)
    {
      auto value       = UnaryRedOp<OP, CODE>::OP::identity;
      auto argred_type = CuNumericRuntime::get_runtime()->get_argred_type(type);
      return Scalar(value, argred_type);
    }

    template <legate::Type::Code CODE, std::enable_if_t<!UnaryRedOp<OP, CODE>::valid>* = nullptr>
    Scalar operator()(const legate::Type&)
    {
      assert(false);
      return Scalar(0);
    }
  };

  template <UnaryRedCode OP>
  Scalar operator()(const legate::Type& type)
  {
    return legate::type_dispatch(type.code(), generator<OP>{}, type);
  }
};

Scalar CuNumericRuntime::get_reduction_identity(UnaryRedCode op, const legate::Type& type)
{
  auto key    = std::make_pair(op, type.code());
  auto finder = identities.find(key);
  if (identities.end() != finder) {
    return finder->second;
  }

  auto identity = op_dispatch(op, generate_identity_fn{}, type);
  identities.insert({key, identity});
  return identity;
}

legate::Type CuNumericRuntime::get_argred_type(const legate::Type& value_type)
{
  auto finder = argred_types_.find(value_type.code());
  if (finder != argred_types_.end()) {
    return finder->second;
  }

  auto argred_type = legate::struct_type({legate::int64(), value_type}, true /*align*/);
  argred_types_.insert({value_type.code(), argred_type});
  return argred_type;
}

namespace {

const std::unordered_map<UnaryRedCode, legate::ReductionOpKind> TO_CORE_REDOP = {
  {UnaryRedCode::ALL, legate::ReductionOpKind::MUL},
  {UnaryRedCode::ANY, legate::ReductionOpKind::ADD},
  {UnaryRedCode::ARGMAX, legate::ReductionOpKind::MAX},
  {UnaryRedCode::ARGMIN, legate::ReductionOpKind::MIN},
  {UnaryRedCode::CONTAINS, legate::ReductionOpKind::ADD},
  {UnaryRedCode::COUNT_NONZERO, legate::ReductionOpKind::ADD},
  {UnaryRedCode::MAX, legate::ReductionOpKind::MAX},
  {UnaryRedCode::MIN, legate::ReductionOpKind::MIN},
  {UnaryRedCode::PROD, legate::ReductionOpKind::MUL},
  {UnaryRedCode::SUM, legate::ReductionOpKind::ADD},
};

}  // namespace

legate::ReductionOpKind CuNumericRuntime::get_reduction_op(UnaryRedCode op)
{
  return TO_CORE_REDOP.at(op);
}

legate::AutoTask CuNumericRuntime::create_task(CuNumericOpCode op_code)
{
  return legate_runtime_->create_task(library_, op_code);
}

void CuNumericRuntime::submit(legate::AutoTask&& task) { legate_runtime_->submit(std::move(task)); }

uint32_t CuNumericRuntime::get_next_random_epoch() { return next_epoch_++; }

/*static*/ CuNumericRuntime* CuNumericRuntime::get_runtime() { return runtime_; }

/*static*/ void CuNumericRuntime::initialize(legate::Runtime* legate_runtime,
                                             legate::Library library)
{
  runtime_ = new CuNumericRuntime(legate_runtime, library);
}

}  // namespace cunumeric
