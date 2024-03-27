/* Copyright 2023 NVIDIA Corporation
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

#include <iomanip>
#include <iostream>
#include <sstream>

#include <gtest/gtest.h>
#include "legate.h"
#include "cunumeric.h"
#include "util.inl"

void bincount_test()
{
  // case: x, no w, min_length=0. out NDArray type is int64_t if no weights
  std::array<int64_t, 7> exp1  = {0, 1, 1, 2, 0, 1, 1};
  std::array<int32_t, 6> in_x1 = {1, 2, 3, 3, 5, 6};
  auto A1                      = cunumeric::zeros({6}, legate::int32());
  assign_values_to_array<int32_t, 1>(A1, in_x1.data(), in_x1.size());
  auto B1 = cunumeric::bincount(A1);
  check_array_eq<int64_t, 1>(B1, exp1.data(), exp1.size());

  // case: x, w, min_length=0.
  std::array<double, 7> exp2  = {0, 1, 1.2, 2, 0, 1, 0.1};
  std::array<double, 6> in_w2 = {1, 1.2, 1, 1, 1, 0.1};
  auto w2                     = cunumeric::zeros({6}, legate::float64());
  assign_values_to_array<double, 1>(w2, in_w2.data(), in_w2.size());
  auto B2 = cunumeric::bincount(A1, w2);
  check_array_eq<double, 1>(B2, exp2.data(), exp2.size());

  // case: x, no w, min_length=8. out NDArray type is int64_t if no weights
  std::array<int64_t, 8> exp3 = {0, 1, 1, 2, 0, 1, 1, 0};
  auto B3                     = cunumeric::bincount(A1, std::nullopt, 8);
  check_array_eq<int64_t, 1>(B3, exp3.data(), exp3.size());

  // case: x of length 1, no w, min_length=0
  std::array<int64_t, 6> exp4 = {0, 0, 0, 0, 0, 1};
  auto A4                     = cunumeric::full({1}, cunumeric::Scalar(5));
  // If we use another way to initialize A4 of length 1 as below, it would rasie error. Seems a lock
  // issue. In this way, if A4 is not of length 1, it pass. int64_t in_x4[1] = {5}; auto A4 =
  // cunumeric::zeros({1}, legate::int64()); assign_values_to_array(A4, (void *)in_x4,
  // sizeof(in_x4)/sizeof(int64_t)); cpp_tests: legion/runtime/realm/runtime_impl.cc:2755:
  // Realm::RegionInstanceImpl* Realm::RuntimeImpl::get_instance_impl(Realm::ID): Assertion `0 &&
  // "invalid instance handle"' failed.
  auto B4 = cunumeric::bincount(A4);
  check_array_eq<int64_t, 1>(B4, exp4.data(), exp4.size());

  // case: x of length 1, w of length 1, min_length=0
  std::array<double, 6> exp5 = {0, 0, 0, 0, 0, 1.3};
  auto w5                    = cunumeric::full({1}, cunumeric::Scalar(1.3));
  auto B5                    = cunumeric::bincount(A4, w5);
  check_array_eq<double, 1>(B5, exp5.data(), exp5.size());

  // case: x of length 1, w of length 1, min_length=8
  std::array<double, 8> exp6 = {0, 0, 0, 0, 0, 1.3, 0, 0};
  auto B6                    = cunumeric::bincount(A4, w5, 8);
  check_array_eq<double, 1>(B6, exp6.data(), exp6.size());
}

void bincount_negative_test()
{
  // case: x.size() == 0
  auto A1 = cunumeric::full({0}, cunumeric::Scalar(5));
  EXPECT_THROW(cunumeric::bincount(A1), std::invalid_argument);

  // case: x.dim() != 1
  auto A2 = cunumeric::full({1, 1}, cunumeric::Scalar(5));
  EXPECT_THROW(cunumeric::bincount(A2), std::invalid_argument);

  // case: x.type() is not int
  auto A3 = cunumeric::full({3}, cunumeric::Scalar(1.3));
  EXPECT_THROW(cunumeric::bincount(A3), std::invalid_argument);

  // case: x.shape() != w.shape()
  auto A4 = cunumeric::zeros({6}, legate::int32());
  auto w4 = cunumeric::zeros({4}, legate::int32());
  EXPECT_THROW(cunumeric::bincount(A4, w4), std::invalid_argument);

  // case: w.type() is not convertible to float64
  auto w5 = cunumeric::zeros({6}, legate::complex64());
  EXPECT_THROW(cunumeric::bincount(A4, w5), std::invalid_argument);

  // case: x is negative
  std::array<int32_t, 6> in_x = {1, 2, -3, 4, 5, 6};
  auto A7                     = cunumeric::zeros({6}, legate::int32());
  assign_values_to_array<int32_t, 1>(A7, in_x.data(), in_x.size());
  EXPECT_THROW(cunumeric::bincount(A7), std::invalid_argument);
}

// void cpp_test()
TEST(Bincount, Normal) { bincount_test(); }

TEST(Bincount, Negative) { bincount_negative_test(); }
