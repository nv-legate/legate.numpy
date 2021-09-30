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

namespace legate {
namespace numpy {

class NumPyRuntime;

class Array {
  friend class NumPyRuntime;

 private:
  Array(NumPyRuntime* runtime,
        LibraryContext* context,
        std::vector<int64_t> shape,
        std::shared_ptr<LogicalStore> store);

 public:
  template <typename T, int32_t DIM>
  AccessorRW<T, DIM> get_accessor();

 public:
  void random(int32_t gen_code);

 private:
  NumPyRuntime* runtime_;
  LibraryContext* context_;
  std::vector<int64_t> shape_;
  std::shared_ptr<LogicalStore> store_;
};

}  // namespace numpy
}  // namespace legate

#include "numpy/array.inl"
