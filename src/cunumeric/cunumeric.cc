/* Copyright 2021-2022 NVIDIA Corporation
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

#include "cunumeric/cunumeric_task.h"
#include "cunumeric/mapper.h"
#include "cunumeric/runtime.h"
#include "cunumeric/unary/unary_red_util.h"

using namespace legate;

namespace cunumeric {

static const char* const cunumeric_library_name = "cunumeric";

/*static*/ TaskRegistrar& CuNumericRegistrar::get_registrar()
{
  static TaskRegistrar registrar;
  return registrar;
}

void registration_callback()
{
  ResourceConfig config;
  config.max_tasks         = CUNUMERIC_MAX_TASKS;
  config.max_reduction_ops = CUNUMERIC_MAX_REDOPS;

  auto library = Runtime::get_runtime()->create_library(
    cunumeric_library_name, config, std::make_unique<CuNumericMapper>());

  CuNumericRegistrar::get_registrar().register_all_tasks(library);
}

void bootstrapping_callback(Legion::Machine machine,
                            Legion::Runtime* legion_runtime,
                            const std::set<Legion::Processor>& local_procs)
{
  registration_callback();

  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(cunumeric_library_name);

  CuNumericRuntime::initialize(runtime, library);
}

}  // namespace cunumeric

extern "C" {

void cunumeric_perform_registration(void)
{
  legate::Core::perform_registration<cunumeric::registration_callback>();
}

bool cunumeric_has_curand()
{
#if defined(LEGATE_USE_CUDA) || defined(CUNUMERIC_CURAND_FOR_CPU_BUILD)
  return true;
#else
  return false;
#endif
}
}
