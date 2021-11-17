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

class Array;

void initialize(int32_t argc, char** argv);

std::shared_ptr<Array> array(std::vector<size_t> shape, legate::LegateTypeCode type);

std::shared_ptr<Array> abs(std::shared_ptr<Array> input);

std::shared_ptr<Array> add(std::shared_ptr<Array> rhs1, std::shared_ptr<Array> rhs2);

std::shared_ptr<Array> dot(std::shared_ptr<Array> rhs1, std::shared_ptr<Array> rhs2);

std::shared_ptr<Array> negative(std::shared_ptr<Array> input);

std::shared_ptr<Array> random(std::vector<size_t> shape);

std::shared_ptr<Array> full(std::vector<size_t> shape, const Scalar& value);

std::shared_ptr<Array> sum(std::shared_ptr<Array> input);

}  // namespace cunumeric
