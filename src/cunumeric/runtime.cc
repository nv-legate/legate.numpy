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
#include "cunumeric/array.h"

namespace cunumeric {

/*static*/ CuNumericRuntime* CuNumericRuntime::runtime_;

extern void bootstrapping_callback(Legion::Machine machine,
                                   Legion::Runtime* runtime,
                                   const std::set<Legion::Processor>& local_procs);

void initialize(int32_t argc, char** argv)
{
  Legion::Runtime::perform_registration_callback(bootstrapping_callback, true /*global*/);
}

CuNumericRuntime::CuNumericRuntime(legate::Runtime* legate_runtime, legate::LibraryContext* context)
  : legate_runtime_(legate_runtime), context_(context)
{
}

std::shared_ptr<Array> CuNumericRuntime::create_array(std::vector<size_t> shape,
                                                      legate::LegateTypeCode type)
{
  // TODO: We need a type system for cuNumeric and should not use the core types
  auto store = legate_runtime_->create_store(shape, type);
  auto array = new Array(this, context_, std::move(store));
  return std::shared_ptr<Array>(array);
}

legate::LogicalStore CuNumericRuntime::create_scalar_store(const Scalar& value)
{
  return legate_runtime_->create_store(value);
}

std::unique_ptr<legate::Task> CuNumericRuntime::create_task(CuNumericOpCode op_code)
{
  return legate_runtime_->create_task(context_, op_code);
}

void CuNumericRuntime::submit(std::unique_ptr<legate::Task> task)
{
  legate_runtime_->submit(std::move(task));
}

uint32_t CuNumericRuntime::get_next_random_epoch() { return next_epoch_++; }

/*static*/ CuNumericRuntime* CuNumericRuntime::get_runtime() { return runtime_; }

/*static*/ void CuNumericRuntime::initialize(legate::Runtime* legate_runtime,
                                             legate::LibraryContext* context)
{
  runtime_ = new CuNumericRuntime(legate_runtime, context);
}

}  // namespace cunumeric
