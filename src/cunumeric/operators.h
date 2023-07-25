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

void initialize(int32_t argc, char** argv);

NDArray array(std::vector<size_t> shape, const legate::Type& type);

NDArray abs(NDArray input);

NDArray add(NDArray rhs1, NDArray rhs2, std::optional<NDArray> out = std::nullopt);

NDArray multiply(NDArray rhs1, NDArray rhs2, std::optional<NDArray> out = std::nullopt);

NDArray dot(NDArray rhs1, NDArray rhs2);

NDArray negative(NDArray input);

NDArray random(std::vector<size_t> shape);

NDArray zeros(std::vector<size_t> shape, std::optional<legate::Type> type = std::nullopt);

NDArray full(std::vector<size_t> shape, const Scalar& value);

NDArray sum(NDArray input);

NDArray unique(NDArray input);

NDArray arange(std::optional<double> start = 0,
               std::optional<double> stop  = std::nullopt,
               std::optional<double> step  = 1,
               const legate::Type& type    = legate::float64());

NDArray as_array(legate::LogicalStore store);

NDArray array_equal(NDArray input0, NDArray input1);

std::vector<NDArray> nonzero(NDArray input);

NDArray eye(size_t n,
            std::optional<size_t> m,
            int32_t k                = 0,
            const legate::Type& type = legate::float64());

NDArray tril(NDArray rhs, int32_t k = 0);

NDArray triu(NDArray rhs, int32_t k = 0);

NDArray bartlett(int64_t M);

NDArray blackman(int64_t M);

NDArray hamming(int64_t M);

NDArray hanning(int64_t M);

NDArray kaiser(int64_t M, double beta);

}  // namespace cunumeric
