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

#include "legion.h"

// #define _USE_MATH_DEFINES

// #include <math.h>
// #include <complex>


namespace cunumeric {

  template <legate::LegateTypeCode CODE>
  struct isnan {
    using T                     = legate::legate_type_of<CODE>;

    isnan() {}

    template <typename _T = T, std::enable_if_t<!std::is_floating_point<_T>::value>* = nullptr>
    inline constexpr bool operator()(const T& x) const
    {
      return false;
    }

    template <typename _T = T, std::enable_if_t<std::is_floating_point<_T>::value>* = nullptr>
    inline constexpr bool operator()(const T& x) const
    {
      using std::isnan;
      return isnan(x);
    }

    template <typename _T>
    inline constexpr bool operator()(const complex<_T>& x) const
    {
      return std::isnan(x.imag()) || std::isnan(x.real());
    }
  };

  // template <>
  // struct isnan<legate::LegateTypeCode::HALF_LT> {
  // using T                     = __half;

  // isnan(const std::vector<legate::Store>& args) {}

  // __CUDA_HD__ bool operator()(const __half& x) const { return isnan(x); }
  // };

}