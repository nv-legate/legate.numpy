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
#include "random/rand_util.h"

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
  // TODO: We need a type system for NumPy and should not use the core types
  auto store = legate_runtime_->create_store(shape, type);
  auto array = new NumPyArray(this, std::move(shape), std::move(store));
  return std::shared_ptr<NumPyArray>(array);
}

uint32_t NumPyRuntime::get_next_random_epoch() { return next_epoch_++; }

/*static*/ NumPyRuntime* NumPyRuntime::get_runtime() { return runtime_; }

/*static*/ void NumPyRuntime::initialize(Runtime* legate_runtime, LibraryContext* context)
{
  runtime_ = new NumPyRuntime(legate_runtime, context);
}

NumPyArray::NumPyArray(NumPyRuntime* runtime,
                       std::vector<int64_t> shape,
                       std::shared_ptr<LogicalStore> store)
  : runtime_(runtime), shape_(std::move(shape)), store_(store)
{
}

void NumPyArray::random(int32_t gen_code)
{
  // auto task = runtime_->create_task(NumPyOpCode::NUMPY_RAND);
  // task->add_output(out);
  // task->add_scalar_arg(Scalar(static_cast<int32_t>(RandGenCode::UNIFORM)));
  // task->add_scalar_arg(Scalar(runtime_->get_next_random_epoch()));
  // auto strides = compute_strides();
  // task.add_scalar_arg(Scalar(static_cast<uint32_t>(strides.size()));
  // for (auto stride : strides) task.add_scalar_arg(Scalar(stride));
  // runtime_->submit(std::move(task));
}

std::shared_ptr<NumPyArray> array(std::vector<int64_t> shape, LegateTypeCode type)
{
  return NumPyRuntime::get_runtime()->create_array(std::move(shape), type);
}

std::shared_ptr<NumPyArray> random(std::vector<int64_t> shape)
{
  auto runtime = NumPyRuntime::get_runtime();
  auto out     = runtime->create_array(std::move(shape), LegateTypeCode::DOUBLE_LT);
  out->random(static_cast<int32_t>(RandGenCode::UNIFORM));
  return out;
}

}  // namespace numpy
}  // namespace legate
