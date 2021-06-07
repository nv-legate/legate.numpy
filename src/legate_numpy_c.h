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

#ifndef __LEGATE_NUMPY_C_H__
#define __LEGATE_NUMPY_C_H__

#include "legate_preamble.h"

// Match these to NumPyOpCode in legate/numpy/config.py
enum NumPyOpCode {
  NUMPY_BINARY_OP        = 400000,
  NUMPY_SCALAR_BINARY_OP = 400002,
  NUMPY_FILL             = 400003,
  NUMPY_SCALAR_UNARY_RED = 400004,
  NUMPY_UNARY_RED        = 400005,
  NUMPY_UNARY_OP         = 400006,
  NUMPY_SCALAR_UNARY_OP  = 400007,
  NUMPY_BINARY_RED       = 400008,
  NUMPY_CONVERT          = 400010,
  NUMPY_SCALAR_CONVERT   = 400011,
  NUMPY_WHERE            = 400012,
  NUMPY_SCALAR_WHERE     = 400013,
  NUMPY_READ             = 400014,
  NUMPY_WRITE            = 400015,
  NUMPY_DIAG             = 400016,
  NUMPY_MATMUL           = 400017,
  NUMPY_MATVECMUL        = 400018,
  NUMPY_DOT              = 400019,
  NUMPY_BINCOUNT         = 400020,
  NUMPY_EYE              = 400021,
  NUMPY_RAND             = 400022,
  NUMPY_ARANGE           = 400023,
  NUMPY_TRANSPOSE        = 400024,
  NUMPY_TILE             = 400025,
};

// Match these to NumPyRedopCode in legate/numpy/config.py
enum NumPyRedopID {
  NUMPY_ARGMIN_REDOP = LEGION_REDOP_KIND_TOTAL + 1,
  NUMPY_ARGMAX_REDOP,
  NUMPY_SCALAR_MAX_REDOP    = 500,
  NUMPY_SCALAR_MIN_REDOP    = 501,
  NUMPY_SCALAR_PROD_REDOP   = 502,
  NUMPY_SCALAR_SUM_REDOP    = 503,
  NUMPY_SCALAR_ARGMAX_REDOP = 504,
  NUMPY_SCALAR_ARGMIN_REDOP = 505,
};

// We provide a global class of projection functions
// Match these to NumPyProjCode in legate/numpy/config.py
enum NumPyProjectionCode {
  // 2D reduction
  NUMPY_PROJ_2D_1D_X = 1,  // keep x
  NUMPY_PROJ_2D_1D_Y = 2,  // keep y
  // 2D broadcast
  NUMPY_PROJ_2D_2D_X = 3,  // keep x
  NUMPY_PROJ_2D_2D_Y = 4,  // keep y
  // 2D promotion
  NUMPY_PROJ_1D_2D_X = 5,  // 1D point becomes (x, 0)
  NUMPY_PROJ_1D_2D_Y = 6,  // 1D point becomes (0, x)
  // 2D transpose
  NUMPY_PROJ_2D_2D_YX = 7,  // transpose (x,y) to (y,x)
  // 3D reduction
  NUMPY_PROJ_3D_2D_XY = 8,   // keep x and y
  NUMPY_PROJ_3D_2D_XZ = 9,   // keep x and z
  NUMPY_PROJ_3D_2D_YZ = 10,  // keep y and z
  NUMPY_PROJ_3D_1D_X  = 11,  // keep x
  NUMPY_PROJ_3D_1D_Y  = 12,  // keep y
  NUMPY_PROJ_3D_1D_Z  = 13,  // keep z
  // 3D broadcast
  NUMPY_PROJ_3D_3D_XY = 14,  // keep x and y, broadcast z
  NUMPY_PROJ_3D_3D_XZ = 15,  // keep x and z, broadcast y
  NUMPY_PROJ_3D_3D_YZ = 16,  // keep y and z, broadcast x
  NUMPY_PROJ_3D_3D_X  = 17,  // keep x, broadcast y and z
  NUMPY_PROJ_3D_3D_Y  = 18,  // keep y, broadcast x and z
  NUMPY_PROJ_3D_3D_Z  = 19,  // keep z, broadcast x and y
  // 3D promotion
  NUMPY_PROJ_2D_3D_XY = 22,  // 2D point becomes (x, y, 0)
  NUMPY_PROJ_2D_3D_XZ = 23,  // 2D point becomes (x, 0, y)
  NUMPY_PROJ_2D_3D_YZ = 24,  // 2D point becomes (0, x, y)
  NUMPY_PROJ_1D_3D_X  = 25,  // 1D point becomes (x, 0, 0)
  NUMPY_PROJ_1D_3D_Y  = 26,  // 1D point becomes (0, x, 0)
  NUMPY_PROJ_1D_3D_Z  = 27,  // 1D point becomes (0, 0, x)
  // Must always be last
  NUMPY_PROJ_LAST = 49,
};

// We provide a global class of sharding functions
enum NumPyShardingCode {
  NUMPY_SHARD_TILE_1D       = 1,
  NUMPY_SHARD_TILE_2D       = 2,
  NUMPY_SHARD_TILE_3D       = 3,
  NUMPY_SHARD_TILE_2D_YX    = 4,  // transpose
  NUMPY_SHARD_TILE_3D_2D_XY = 5,
  NUMPY_SHARD_TILE_3D_2D_XZ = 6,
  NUMPY_SHARD_TILE_3D_2D_YZ = 7,
  NUMPY_SHARD_EXTRA         = 91,
  // Leave space for some extra IDs for transform sharding functions
  NUMPY_SHARD_LAST = 1024,  // This one must be last
};

// Match these to NumPyMappingTag in legate/numpy/config.py
enum NumPyTag {
  NUMPY_SUBRANKABLE_TAG = 0x1,
  NUMPY_CPU_ONLY_TAG    = 0x2,
  NUMPY_GPU_ONLY_TAG    = 0x4,
  NUMPY_NO_MEMOIZE_TAG  = 0x8,
  NUMPY_KEY_REGION_TAG  = 0x10,
};

// Match these to NumPyTunable in legate/numpy/config.py
enum NumPyTunable {
  NUMPY_TUNABLE_NUM_PIECES            = 1,
  NUMPY_TUNABLE_NUM_GPUS              = 2,
  NUMPY_TUNABLE_TOTAL_NODES           = 3,
  NUMPY_TUNABLE_LOCAL_CPUS            = 4,
  NUMPY_TUNABLE_LOCAL_GPUS            = 5,
  NUMPY_TUNABLE_LOCAL_OPENMPS         = 6,
  NUMPY_TUNABLE_MIN_SHARD_VOLUME      = 8,
  NUMPY_TUNABLE_MAX_EAGER_VOLUME      = 9,
  NUMPY_TUNABLE_FIELD_REUSE_SIZE      = 10,
  NUMPY_TUNABLE_FIELD_REUSE_FREQUENCY = 11,
};

enum NumPyBounds {
  NUMPY_MAX_MAPPERS = 1,
  NUMPY_MAX_REDOPS  = 1024,
  NUMPY_MAX_TASKS   = 1048576,
};

#ifdef __cplusplus
extern "C" {
#endif

void legate_numpy_perform_registration();

void legate_numpy_create_transform_sharding_functor(
  unsigned first, unsigned offset, unsigned M, unsigned N, const long* transform);

#ifdef __cplusplus
}
#endif

#endif  // __LEGATE_NUMPY_C_H__
