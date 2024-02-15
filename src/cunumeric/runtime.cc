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
#include "cunumeric/unary/unary_red_util.h"

namespace cunumeric {

/*static*/ CuNumericRuntime* CuNumericRuntime::runtime_;

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

NDArray CuNumericRuntime::create_array(std::vector<uint64_t> shape,
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
