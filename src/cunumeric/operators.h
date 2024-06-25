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
#include <optional>

#include "legate.h"
#include "cunumeric/ndarray.h"
#include "cunumeric/typedefs.h"

namespace cunumeric {

legate::Logger& cunumeric_log();

void initialize(int32_t argc, char** argv);

NDArray array(std::vector<uint64_t> shape, const legate::Type& type);

NDArray abs(NDArray input);

NDArray add(NDArray rhs1, NDArray rhs2, std::optional<NDArray> out = std::nullopt);

NDArray multiply(NDArray rhs1, NDArray rhs2, std::optional<NDArray> out = std::nullopt);

NDArray dot(NDArray rhs1, NDArray rhs2);

NDArray negative(NDArray input);

NDArray random(std::vector<uint64_t> shape);

NDArray zeros(std::vector<uint64_t> shape, std::optional<legate::Type> type = std::nullopt);

NDArray full(std::vector<uint64_t> shape, const Scalar& value);

NDArray all(NDArray input,
            std::optional<std::vector<int32_t>> axis = std::nullopt,
            std::optional<NDArray> out               = std::nullopt,
            std::optional<bool> keepdims             = std::nullopt,
            std::optional<NDArray> where             = std::nullopt);

NDArray sum(NDArray input);

NDArray amax(NDArray input);

NDArray amin(NDArray input);

NDArray unique(NDArray input);

NDArray swapaxes(NDArray input, int32_t axis1, int32_t axis2);

template <typename T>
NDArray arange(T start, std::optional<T> stop = std::nullopt, T step = 1);

NDArray arange(Scalar start, Scalar stop = legate::Scalar{}, Scalar step = legate::Scalar{});

NDArray as_array(legate::LogicalStore store);

NDArray array_equal(NDArray input0, NDArray input1);

std::vector<NDArray> nonzero(NDArray input);

NDArray eye(int32_t n,
            std::optional<int32_t> m = std::nullopt,
            int32_t k                = 0,
            const legate::Type& type = legate::float64());

NDArray tril(NDArray rhs, int32_t k = 0);

NDArray triu(NDArray rhs, int32_t k = 0);

NDArray bartlett(int64_t M);

NDArray blackman(int64_t M);

NDArray hamming(int64_t M);

NDArray hanning(int64_t M);

NDArray kaiser(int64_t M, double beta);

NDArray bincount(NDArray x, std::optional<NDArray> weights = std::nullopt, uint32_t min_length = 0);

NDArray convolve(NDArray a, NDArray v);

NDArray sort(NDArray input, std::optional<int32_t> axis = -1, std::string kind = "quicksort");

NDArray argsort(NDArray input, std::optional<int32_t> axis = -1, std::string kind = "quicksort");

NDArray msort(NDArray input);

NDArray sort_complex(NDArray input);

NDArray transpose(NDArray a);

NDArray transpose(NDArray a, std::vector<int32_t> axes);

NDArray moveaxis(NDArray a, std::vector<int32_t> source, std::vector<int32_t> destination);

NDArray argwhere(NDArray input);

NDArray flip(NDArray input, std::optional<std::vector<int32_t>> axis = std::nullopt);

void put(NDArray& a, NDArray indices, NDArray values, std::string mode = "raise");

// helper methods
int32_t normalize_axis_index(int32_t axis, int32_t ndim);

std::vector<int32_t> normalize_axis_vector(std::vector<int32_t> axis,
                                           int32_t ndim,
                                           bool allow_duplicate = false);

NDArray diag(NDArray v, int32_t k = 0);

NDArray diagonal(NDArray a,
                 int32_t offset               = 0,
                 std::optional<int32_t> axis1 = std::nullopt,
                 std::optional<int32_t> axis2 = std::nullopt,
                 std::optional<bool> extract  = std::nullopt);

NDArray trace(NDArray a,
              int32_t offset                   = 0,
              int32_t axis1                    = 0,
              int32_t axis2                    = 1,
              std::optional<legate::Type> type = std::nullopt,
              std::optional<NDArray> out       = std::nullopt);
}  // namespace cunumeric
#include "cunumeric/operators.inl"