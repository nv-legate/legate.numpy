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

namespace legate {
namespace numpy {

template <typename T, int32_t DIM>
AccessorRW<T, DIM> Array::get_accessor()
{
  auto mapped = store_->get_physical_store(context_);
  auto shape  = mapped->shape<DIM>();
  return mapped->read_write_accessor<T, DIM>(shape);
}

}  // namespace numpy
}  // namespace legate
