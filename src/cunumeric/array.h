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
#include "cunumeric/typedefs.h"

namespace cunumeric {

class CuNumericRuntime;

class Array {
  friend class CuNumericRuntime;

 private:
  Array(CuNumericRuntime* runtime, legate::LibraryContext* context, legate::LogicalStore store);

 public:
  int32_t dim() const;
  const std::vector<size_t>& shape() const;
  legate::LegateTypeCode code() const;

 public:
  template <typename T, int32_t DIM>
  legate::AccessorRW<T, DIM> get_accessor();

 public:
  void random(int32_t gen_code);
  void fill(const Scalar& value, bool argval);
  void binary_op(int32_t op_code, std::shared_ptr<Array> rhs1, std::shared_ptr<Array> rhs2);
  void unary_op(int32_t op_code, std::shared_ptr<Array> input);

 private:
  CuNumericRuntime* runtime_;
  legate::LibraryContext* context_;
  legate::LogicalStore store_;
};

}  // namespace cunumeric

#include "cunumeric/array.inl"
