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

namespace {
template <typename T>
std::string to_string_1d(legate::AccessorRO<T, 1> acc, const std::vector<size_t>& shape)
{
  std::stringstream ss;

  ss << "[";
  for (auto i = 0; i < shape[0]; ++i) {
    if (i > 0) ss << ", ";
    ss << std::setw(9) << std::setprecision(6) << acc[i];
  }
  ss << "]";

  return ss.str();
}

template <typename T>
std::string to_string_2d(legate::AccessorRO<T, 2> acc, const std::vector<size_t>& shape)
{
  std::stringstream ss;

  ss << "[";
  for (auto i = 0; i < shape[0]; ++i) {
    if (i > 0) ss << ",\n ";
    ss << "[";
    for (auto j = 0; j < shape[1]; ++j) {
      if (j > 0) ss << ", ";
      ss << std::setw(9) << std::setprecision(3) << acc[i][j];
    }
    ss << "]";
  }

  return ss.str();
}

template <typename T>
std::string to_string_3d(legate::AccessorRO<T, 3> acc, const std::vector<size_t>& shape)
{
  std::stringstream ss;

  ss << "[";
  for (auto k = 0; k < shape[0]; ++k) {
    if (k > 0) ss << ",\n ";
    ss << "[";
    for (auto i = 0; i < shape[1]; ++i) {
      if (i > 0) ss << ",\n ";
      ss << "[";
      for (auto j = 0; j < shape[2]; ++j) {
        if (j > 0) ss << ", ";
        ss << std::setw(9) << std::setprecision(3) << acc[k][i][j];
      }
      ss << "]";
    }
    ss << "]";
  }
  ss << "]";

  return ss.str();
}

template <typename T>
std::string check_array_eq_1d(legate::AccessorRO<T, 1> acc,
                              T* values_ptr,
                              const std::vector<size_t>& shape)
{
  std::stringstream ss;

  ss << "[";
  for (auto i = 0; i < shape[0]; ++i) {
    if (i > 0) ss << ", ";
    ss << std::setw(9) << std::setprecision(6) << acc[i];
    EXPECT_EQ(acc[i], values_ptr[i]);
  }
  ss << "]";

  return ss.str();
}

template <typename T>
std::string check_array_eq_2d(legate::AccessorRO<T, 2> acc,
                              T* values_ptr,
                              const std::vector<size_t>& shape)
{
  std::stringstream ss;

  ss << "[";
  for (auto i = 0; i < shape[0]; ++i) {
    if (i > 0) ss << ",\n ";
    ss << "[";
    for (auto j = 0; j < shape[1]; ++j) {
      if (j > 0) ss << ", ";
      ss << std::setw(9) << std::setprecision(3) << acc[i][j];
      EXPECT_EQ(acc[i][j], values_ptr[i * shape[1] + j]);
    }
    ss << "]";
  }
  ss << "]";

  return ss.str();
}

template <typename T>
std::string check_array_eq_3d(legate::AccessorRO<T, 3> acc,
                              T* values_ptr,
                              const std::vector<size_t>& shape)
{
  std::stringstream ss;

  ss << "[";
  for (auto k = 0; k < shape[0]; ++k) {
    if (k > 0) ss << ",\n ";
    ss << "[";
    for (auto i = 0; i < shape[1]; ++i) {
      if (i > 0) ss << ",\n ";
      ss << "[";
      for (auto j = 0; j < shape[2]; ++j) {
        if (j > 0) ss << ", ";
        ss << std::setw(9) << std::setprecision(3) << acc[k][i][j];
        EXPECT_EQ(acc[k][i][j], values_ptr[k * shape[1] * shape[2] + i * shape[2] + j]);
      }
      ss << "]";
    }
    ss << "]";
  }
  ss << "]";

  return ss.str();
}

template <typename T, int32_t DIM>
struct print_fn;

template <typename T>
struct print_fn<T, 1> {
  void operator()(legate::AccessorRO<T, 1> acc, const std::vector<size_t>& shape)
  {
    std::cerr << to_string_1d<T>(acc, shape) << std::endl;
  }
};

template <typename T>
struct print_fn<T, 2> {
  void operator()(legate::AccessorRO<T, 2> acc, const std::vector<size_t>& shape)
  {
    std::cerr << to_string_2d<T>(acc, shape) << std::endl;
  }
};

template <typename T>
struct print_fn<T, 3> {
  void operator()(legate::AccessorRO<T, 3> acc, const std::vector<size_t>& shape)
  {
    std::cerr << to_string_3d<T>(acc, shape) << std::endl;
  }
};

template <typename T, int32_t DIM>
struct check_array_eq_fn;

template <typename T>
struct check_array_eq_fn<T, 1> {
  void operator()(legate::AccessorRO<T, 1> acc, T* values_ptr, const std::vector<size_t>& shape)
  {
    std::cerr << check_array_eq_1d<T>(acc, values_ptr, shape) << std::endl;
  }
};

template <typename T>
struct check_array_eq_fn<T, 2> {
  void operator()(legate::AccessorRO<T, 2> acc, T* values_ptr, const std::vector<size_t>& shape)
  {
    std::cerr << check_array_eq_2d<T>(acc, values_ptr, shape) << std::endl;
  }
};

template <typename T>
struct check_array_eq_fn<T, 3> {
  void operator()(legate::AccessorRO<T, 3> acc, T* values_ptr, const std::vector<size_t>& shape)
  {
    std::cerr << check_array_eq_3d<T>(acc, values_ptr, shape) << std::endl;
  }
};

template <typename T, int32_t DIM>
struct assign_array_fn;

template <typename T>
struct assign_array_fn<T, 1> {
  void operator()(legate::AccessorWO<T, 1> acc, T* values_ptr, const std::vector<size_t>& shape)
  {
    for (auto i = 0; i < shape[0]; ++i) { acc[i] = values_ptr[i]; }
  }
};

template <typename T>
struct assign_array_fn<T, 2> {
  void operator()(legate::AccessorWO<T, 2> acc, T* values_ptr, const std::vector<size_t>& shape)
  {
    for (auto i = 0; i < shape[0]; ++i) {
      for (auto j = 0; j < shape[1]; ++j) { acc[i][j] = values_ptr[i * shape[1] + j]; }
    }
  }
};

template <typename T>
struct assign_array_fn<T, 3> {
  void operator()(legate::AccessorWO<T, 3> acc, T* values_ptr, const std::vector<size_t>& shape)
  {
    for (auto i = 0; i < shape[0]; ++i) {
      for (auto j = 0; j < shape[1]; ++j) {
        for (auto k = 0; k < shape[2]; ++k) {
          acc[i][j][k] = values_ptr[i * shape[1] * shape[2] + j * shape[2] + k];
        }
      }
    }
  }
};

template <typename T, int32_t DIM>
void print_array(cunumeric::NDArray array)
{
  auto acc    = array.get_read_accessor<T, DIM>();
  auto& shape = array.shape();
  print_fn<T, DIM>()(acc, shape);
}

template <typename T, int32_t DIM>
void check_array_eq(cunumeric::NDArray array, T* values_ptr, size_t length)
{
  assert(array.size() == length);
  auto acc    = array.get_read_accessor<T, DIM>();
  auto& shape = array.shape();
  check_array_eq_fn<T, DIM>()(acc, values_ptr, shape);
}

template <typename T, int32_t DIM>
void assign_values_to_array(cunumeric::NDArray array, T* values_ptr, size_t length)
{
  assert(array.size() == length);
  auto acc    = array.get_write_accessor<T, DIM>();
  auto& shape = array.shape();
  assign_array_fn<T, DIM>()(acc, values_ptr, shape);
}
}  // namespace
