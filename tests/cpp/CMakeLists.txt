#=============================================================================
# Copyright 2023 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================

cmake_minimum_required(VERSION 3.22.1 FATAL_ERROR)

project(cpp_tests VERSION 0.1 LANGUAGES C CXX)

if (NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.13.0.zip
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

find_package(legate_core REQUIRED)
find_package(cunumeric REQUIRED)

enable_testing()

file(GLOB main_SRC ${PROJECT_SOURCE_DIR}/main.cc)
file(GLOB integration_SRC ${PROJECT_SOURCE_DIR}/integration/*.cc)
file(GLOB tasks_SRC ${PROJECT_SOURCE_DIR}/integration/tasks/*.cc)
file(GLOB unit_SRC ${PROJECT_SOURCE_DIR}/unit/*.cc)

if(Legion_USE_CUDA)

find_package(CUDAToolkit REQUIRED)
enable_language(CUDA)

file(GLOB integration_GPU_SRC ${PROJECT_SOURCE_DIR}/integration/*.cu)
list(APPEND integration_SRC ${integration_GPU_SRC})

if(NOT EXISTS ${CMAKE_BINARY_DIR}/RAPIDS.cmake)
  file(DOWNLOAD https://raw.githubusercontent.com/rapidsai/rapids-cmake/branch-23.04/RAPIDS.cmake
       ${CMAKE_BINARY_DIR}/RAPIDS.cmake)
endif()
include(${CMAKE_BINARY_DIR}/RAPIDS.cmake)
include(rapids-find)
include(cmake/thirdparty/get_nccl.cmake)

endif()

add_executable(cpp_tests ${main_SRC} ${tasks_SRC} ${integration_SRC} ${unit_SRC})

target_link_libraries(cpp_tests PRIVATE legate::core PRIVATE cunumeric::cunumeric PRIVATE GTest::gtest)
if(Legion_USE_CUDA)
target_link_libraries(cpp_tests PRIVATE NCCL::NCCL)
endif()

include(GoogleTest)
gtest_discover_tests(cpp_tests DISCOVERY_MODE PRE_TEST WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

include(CPack)
include(GNUInstallDirs)

install(TARGETS cpp_tests DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/cmake-install")
install( TARGETS cpp_tests DESTINATION ${CMAKE_INSTALL_BINDIR})
