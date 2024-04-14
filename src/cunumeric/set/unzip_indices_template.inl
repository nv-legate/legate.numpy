/* Copyright 2022 NVIDIA Corporation
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

// Useful for IDEs
#include "cunumeric/set/unzip_indices.h"
#include "cunumeric/pitches.h"
#include "cunumeric/set/zip_indices.h"

#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>
#include <iostream>

namespace cunumeric {

using namespace legate;

template <typename VAL>
struct ValExtract {
  VAL operator()(const ZippedIndex<VAL>& x) { return x.value; }
};

template <typename VAL>
struct IndexExtract {
  int64_t operator()(const ZippedIndex<VAL>& x) { return x.index; }
};

template <typename exe_pol_t>
struct UnzipIndicesImpl {
  template <Type::Code CODE>
  void operator()(std::vector<Array>& outputs, Array& input, const exe_pol_t& exe_pol)
  {
    using VAL = legate_type_of<CODE>;

    auto input_shape = input.shape<1>();
    size_t res_size  = input_shape.hi[0] - input_shape.lo[0] + 1;

    auto value         = outputs[0].create_output_buffer<VAL, 1>(res_size, true);
    auto index         = outputs[1].create_output_buffer<int64_t, 1>(res_size, true);
    VAL* value_ptr     = value.ptr(0);
    int64_t* index_ptr = index.ptr(0);

    size_t strides[1];
    const ZippedIndex<VAL>* in_ptr =
      input.read_accessor<ZippedIndex<VAL>, 1>(input_shape).ptr(input_shape, strides);
    // unique_reduce has this check, so it's probably worthwhile to keep it here
    assert(input_shape.volume() <= 1 || strides[0] == 1);

    thrust::transform(exe_pol, in_ptr, in_ptr + res_size, value_ptr, ValExtract<VAL>());
    thrust::transform(exe_pol, in_ptr, in_ptr + res_size, index_ptr, IndexExtract<VAL>());
  }
};

template <typename exe_pol_t>
static void unzip_indices_template(TaskContext& context, const exe_pol_t& exe_pol)
{
  auto& input   = context.inputs()[0];
  auto& outputs = context.outputs();

  Type::Code code{input.code()};
  assert(Type::Code::STRUCT == code);
  auto& field_type = static_cast<const StructType&>(input.type()).field_type(0);
  code             = field_type.code;

  type_dispatch(code, UnzipIndicesImpl<exe_pol_t>{}, outputs, input, exe_pol);
}

}  // namespace cunumeric
