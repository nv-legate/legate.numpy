/* Copyright 2024 NVIDIA Corporation
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

#include "cunumeric/cunumeric_task.h"

#include <thrust/functional.h>

namespace cunumeric {

enum class ScanCode : int {
  PROD = CUNUMERIC_SCAN_PROD,
  SUM  = CUNUMERIC_SCAN_SUM,
};

template <typename Functor, typename... Fnargs>
constexpr decltype(auto) op_dispatch(ScanCode op_code, Functor f, Fnargs&&... args)
{
  switch (op_code) {
    case ScanCode::PROD:
      return f.template operator()<ScanCode::PROD>(std::forward<Fnargs>(args)...);
    case ScanCode::SUM: return f.template operator()<ScanCode::SUM>(std::forward<Fnargs>(args)...);
    default: break;
  }
  assert(false);
  return f.template operator()<ScanCode::SUM>(std::forward<Fnargs>(args)...);
}

template <typename Functor, typename... Fnargs>
constexpr decltype(auto) op_dispatch(ScanCode op_code,
                                     bool nan_to_identity,
                                     Functor f,
                                     Fnargs&&... args)
{
  switch (op_code) {
    case ScanCode::PROD:
      if (nan_to_identity) {
        return f.template operator()<ScanCode::PROD, true>(std::forward<Fnargs>(args)...);
      } else {
        return f.template operator()<ScanCode::PROD, false>(std::forward<Fnargs>(args)...);
      }
    case ScanCode::SUM:
      if (nan_to_identity) {
        return f.template operator()<ScanCode::SUM, true>(std::forward<Fnargs>(args)...);
      } else {
        return f.template operator()<ScanCode::SUM, false>(std::forward<Fnargs>(args)...);
      }
    default: break;
  }
  assert(false);
  return f.template operator()<ScanCode::SUM, false>(std::forward<Fnargs>(args)...);
}

template <ScanCode OP_CODE, legate::Type::Code CODE>
struct ScanOp;

template <legate::Type::Code CODE>
struct ScanOp<ScanCode::SUM, CODE> : thrust::plus<legate::type_of<CODE>> {
  using T                           = legate::type_of<CODE>;
  static constexpr int nan_identity = 0;
};

template <legate::Type::Code CODE>
struct ScanOp<ScanCode::PROD, CODE> : thrust::multiplies<legate::type_of<CODE>> {
  using T                           = legate::type_of<CODE>;
  static constexpr int nan_identity = 1;
};

template <>
struct ScanOp<ScanCode::PROD, legate::Type::Code::BOOL> {
  using T                            = bool;
  static constexpr bool nan_identity = true;
  constexpr T operator()(const bool& lhs, const bool& rhs) const { return lhs && rhs; }
};

}  // namespace cunumeric
