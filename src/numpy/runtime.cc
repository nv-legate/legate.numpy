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

#include "numpy/runtime.h"
#include "numpy/array.h"

namespace legate {
namespace numpy {

/*static*/ NumPyRuntime* NumPyRuntime::runtime_;

extern void bootstrapping_callback(Legion::Machine machine,
                                   Legion::Runtime* runtime,
                                   const std::set<Legion::Processor>& local_procs);

void initialize(int32_t argc, char** argv)
{
  Legion::Runtime::perform_registration_callback(bootstrapping_callback, true /*global*/);
}

NumPyRuntime::NumPyRuntime(Runtime* legate_runtime, LibraryContext* context)
  : legate_runtime_(legate_runtime), context_(context)
{
}

std::shared_ptr<Array> NumPyRuntime::create_array(std::vector<int64_t> shape, LegateTypeCode type)
{
  // TODO: We need a type system for NumPy and should not use the core types
  auto store = legate_runtime_->create_store(shape, type);
  auto array = new Array(this, std::move(shape), std::move(store));
  return std::shared_ptr<Array>(array);
}

std::unique_ptr<Task> NumPyRuntime::create_task(NumPyOpCode op_code)
{
  return legate_runtime_->create_task(context_, op_code);
}

void NumPyRuntime::submit(std::unique_ptr<Task> task) { legate_runtime_->submit(std::move(task)); }

uint32_t NumPyRuntime::get_next_random_epoch() { return next_epoch_++; }

/*static*/ NumPyRuntime* NumPyRuntime::get_runtime() { return runtime_; }

/*static*/ void NumPyRuntime::initialize(Runtime* legate_runtime, LibraryContext* context)
{
  runtime_ = new NumPyRuntime(legate_runtime, context);
}

}  // namespace numpy
}  // namespace legate
