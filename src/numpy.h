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

#pragma once

#include <memory>

#include "legate.h"

#include "legate_numpy_c.h"

namespace legate {

namespace numpy {

// TODO: The following functions and classes will later be reorganized into separate header files

void initialize(int32_t argc, char** argv);

class NumPyArray;

class NumPyRuntime {
 private:
  NumPyRuntime(Runtime* legate_runtime, LibraryContext* context);

 public:
  std::shared_ptr<NumPyArray> create_array(std::vector<int64_t> shape, LegateTypeCode type);

 public:
  std::unique_ptr<Task> create_task(NumPyOpCode op_code);
  void submit(std::unique_ptr<Task> task);

 public:
  uint32_t get_next_random_epoch();

 public:
  static NumPyRuntime* get_runtime();
  static void initialize(Runtime* legate_runtime, LibraryContext* context);

 private:
  static NumPyRuntime* runtime_;

 private:
  Runtime* legate_runtime_;
  LibraryContext* context_;
  uint32_t next_epoch_{0};
};

class NumPyArray {
  friend class NumPyRuntime;

 private:
  NumPyArray(NumPyRuntime* runtime,
             std::vector<int64_t> shape,
             std::shared_ptr<LogicalStore> store);

 public:
  void random(int32_t gen_code);

 private:
  NumPyRuntime* runtime_;
  std::vector<int64_t> shape_;
  std::shared_ptr<LogicalStore> store_;
};

std::shared_ptr<NumPyArray> array(std::vector<int64_t> shape, LegateTypeCode type);

std::shared_ptr<NumPyArray> random(std::vector<int64_t> shape);

}  // namespace numpy
}  // namespace legate
