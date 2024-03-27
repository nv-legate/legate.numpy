/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <gtest/gtest.h>
#include "legate.h"
#include "cunumeric.h"

class Environment : public ::testing::Environment {
 public:
  Environment(int argc, char** argv) : argc_(argc), argv_(argv) {}

  void SetUp() override
  {
    EXPECT_EQ(legate::start(argc_, argv_), 0);
    cunumeric::initialize(argc_, argv_);
  }
  void TearDown() override { EXPECT_EQ(legate::finish(), 0); }

 private:
  int argc_;
  char** argv_;
};

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new Environment(argc, argv));

  return RUN_ALL_TESTS();
}
