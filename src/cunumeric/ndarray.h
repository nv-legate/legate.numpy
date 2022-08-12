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

class NDArray {
  friend class CuNumericRuntime;

 private:
  NDArray(legate::LogicalStore&& store);

 public:
  NDArray(const NDArray&)            = default;
  NDArray& operator=(const NDArray&) = default;

 public:
  NDArray(NDArray&&)            = default;
  NDArray& operator=(NDArray&&) = default;

 public:
  int32_t dim() const;
  const std::vector<size_t>& shape() const;
  legate::LegateTypeCode code() const;

 public:
  template <typename T, int32_t DIM>
  legate::AccessorRW<T, DIM> get_accessor();

 public:
  NDArray operator+(const NDArray& other) const;
  NDArray& operator+=(const NDArray& other);

 public:
  void random(int32_t gen_code);
  void fill(const Scalar& value, bool argval);
  void binary_op(int32_t op_code, NDArray rhs1, NDArray rhs2);
  void unary_op(int32_t op_code, NDArray input);
  void unary_reduction(int32_t op_code, NDArray input);
  void fill(NDArray fill_value);
  void dot(NDArray rhs1, NDArray rhs2);

 private:
  legate::LogicalStore broadcast(const std::vector<size_t>& shape, legate::LogicalStore& store);

 public:
  static legate::LibraryContext* get_context();

 private:
  legate::LogicalStore store_;
};

}  // namespace cunumeric

#include "cunumeric/ndarray.inl"
