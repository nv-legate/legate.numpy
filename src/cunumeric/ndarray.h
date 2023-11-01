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
#include <initializer_list>

#include "legate.h"
#include "cunumeric/slice.h"
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
  size_t size() const;
  legate::Type type() const;

 public:
  template <typename T, int32_t DIM>
  legate::AccessorRO<T, DIM> get_read_accessor();
  template <typename T, int32_t DIM>
  legate::AccessorWO<T, DIM> get_write_accessor();

 public:
  NDArray operator+(const NDArray& other) const;
  NDArray operator+(const legate::Scalar& other) const;
  NDArray& operator+=(const NDArray& other);
  NDArray operator*(const NDArray& other) const;
  NDArray operator*(const legate::Scalar& other) const;
  NDArray& operator*=(const NDArray& other);
  NDArray operator[](std::initializer_list<slice> slices) const;
  operator bool() const;

 public:
  // Copy the contents of the other ndarray to this one
  void assign(const NDArray& other);
  void assign(const legate::Scalar& other);

 public:
  void random(int32_t gen_code);
  void fill(const Scalar& value, bool argval);
  void binary_op(int32_t op_code, NDArray rhs1, NDArray rhs2);
  void binary_reduction(int32_t op_code, NDArray rhs1, NDArray rhs2);
  void unary_op(int32_t op_code, NDArray input);
  void unary_reduction(int32_t op_code, NDArray input);
  void fill(NDArray fill_value);
  void eye(int32_t k);
  void trilu(NDArray rhs, int32_t k, bool lower);
  void dot(NDArray rhs1, NDArray rhs2);
  void arange(double start, double stop, double step);
  std::vector<NDArray> nonzero();
  NDArray unique();
  NDArray swapaxes(int32_t axis1, int32_t axis2);
  void create_window(int32_t op_code, int64_t M, std::vector<double> args);
  void bincount(NDArray rhs, std::optional<NDArray> weights = std::nullopt);
  void convolve(NDArray input, NDArray filter);
  void sort(NDArray rhs,
            bool argsort                = false,
            std::optional<int32_t> axis = -1,
            std::string kind            = "quicksort");
  NDArray transpose(std::optional<std::vector<int32_t>> axes = std::nullopt);

 public:
  NDArray as_type(const legate::Type& type);
  legate::LogicalStore get_store();
  int32_t normalize_axis_index(int32_t axis);
  void sort(NDArray rhs, bool argsort, std::optional<int32_t> axis = -1, bool stable = false);

 private:
  legate::LogicalStore broadcast(const std::vector<size_t>& shape, legate::LogicalStore& store);
  legate::LogicalStore broadcast(NDArray rhs1, NDArray rhs2);
  void sort_task(NDArray rhs, bool argsort, bool stable);
  void sort_swapped(NDArray rhs, bool argsort, int32_t sort_axis, bool stable);

 public:
  static legate::Library get_library();

 private:
  legate::LogicalStore store_;
};

}  // namespace cunumeric

#include "cunumeric/ndarray.inl"
