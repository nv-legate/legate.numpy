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

#include "cunumeric/cunumeric_c.h"
#include "cunumeric/typedefs.h"
#include "cunumeric/unary/unary_red_util.h"

namespace cunumeric {

class NDArray;

class CuNumericRuntime {
 private:
  CuNumericRuntime(legate::Runtime* legate_runtime, legate::LibraryContext* context);

 public:
  NDArray create_array(std::unique_ptr<legate::Type> type);
  NDArray create_array(const legate::Type& type);
  NDArray create_array(std::vector<size_t> shape, std::unique_ptr<legate::Type> type);
  NDArray create_array(std::vector<size_t> shape, const legate::Type& type);
  legate::LogicalStore create_scalar_store(const Scalar& value);

 public:
  Scalar get_reduction_identity(UnaryRedCode op, const legate::Type& type);
  Legion::ReductionOpID get_reduction_op(UnaryRedCode op, const legate::Type& type);

 public:
  const legate::Type& get_argred_type(const legate::Type& value_type);

 public:
  std::unique_ptr<legate::AutoTask> create_task(CuNumericOpCode op_code);
  void submit(std::unique_ptr<legate::Task> task);

 public:
  uint32_t get_next_random_epoch();

 public:
  legate::LibraryContext* get_context() const { return context_; }

 public:
  static CuNumericRuntime* get_runtime();
  static void initialize(legate::Runtime* legate_runtime, legate::LibraryContext* context);

 private:
  static CuNumericRuntime* runtime_;

 private:
  legate::Runtime* legate_runtime_;
  legate::LibraryContext* context_;
  uint32_t next_epoch_{0};
  std::unordered_map<legate::Type::Code, std::unique_ptr<legate::Type>> argred_types_;
};

}  // namespace cunumeric
