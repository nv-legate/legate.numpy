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

#include "numpy.h"

#include "legate_numpy_c.h"

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

std::shared_ptr<NumPyArray> NumPyRuntime::create_array(std::vector<int64_t> shape,
                                                       LegateTypeCode type)
{
  // auto store = legate_runtime_->create_store(shape, type);
  auto array = new NumPyArray(this, nullptr);
  return std::shared_ptr<NumPyArray>(array);
}

/*static*/ NumPyRuntime* NumPyRuntime::get_runtime() { return runtime_; }

/*static*/ void NumPyRuntime::initialize(Runtime* legate_runtime, LibraryContext* context)
{
  runtime_ = new NumPyRuntime(legate_runtime, context);
}

NumPyArray::NumPyArray(NumPyRuntime* runtime, std::shared_ptr<LogicalStore> store)
  : runtime_(runtime), store_(store)
{
}

std::shared_ptr<NumPyArray> array(std::vector<int64_t> shape, LegateTypeCode type)
{
  return NumPyRuntime::get_runtime()->create_array(std::move(shape), type);
}

std::shared_ptr<NumPyArray> arange(int64_t stop) { return nullptr; }

}  // namespace numpy
}  // namespace legate
