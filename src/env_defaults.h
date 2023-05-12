/* Copyright 2021-2022 NVIDIA Corporation
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

// These values are copied manually in cunumeric.settings and there is a Python
// unit test that will maintain that these values and the Python settings
// values agree. If these values are modified, the corresponding Python values
// must also be updated.

// 1 << 16 (need actual number for python to parse)
#define MIN_GPU_CHUNK_DEFAULT 65536
#define MIN_GPU_CHUNK_TEST 2

// 1 << 10 (need actual number for python to parse)
#define MIN_CPU_CHUNK_DEFAULT 1024
#define MIN_CPU_CHUNK_TEST 2

// 1 << 13 (need actual number for python to parse)
#define MIN_OMP_CHUNK_DEFAULT 8192
#define MIN_OMP_CHUNK_TEST 2
