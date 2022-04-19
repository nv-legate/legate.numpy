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

#include "cunumeric/random/bitgenerator.h"
#include "cunumeric/random/bitgenerator_template.inl"
#include "cunumeric/random/bitgenerator_util.h"

#include "cunumeric/cuda_help.h"
#include "cunumeric/random/curand_help.h"

#include "cunumeric/random/bitgenerator_curand.inl"

namespace cunumeric {

using namespace Legion;
using namespace legate;

struct GPUGenerator : public CURANDGenerator {
  GPUGenerator(BitGeneratorType gentype)
  {
    cudaStream_t stream;
    CHECK_CUDA(::cudaStreamCreate(&stream));
    CHECK_CURAND(::curandCreateGenerator(&gen, get_curandRngType(gentype)));
    CHECK_CURAND(::curandSetStream(gen, stream));
    offset             = 0;
    type               = get_curandRngType(gentype);
    supports_skipahead = supportsSkipAhead(type);
    dev_buffer_size    = DEFAULT_DEV_BUFFER_SIZE;
// TODO: use realm allocator
#if (__CUDACC_VER_MAJOR__ > 11 || ((__CUDACC_VER_MAJOR__ >= 11) && (__CUDACC_VER_MINOR__ >= 2)))
    CHECK_CUDA(::cudaMallocAsync(&(dev_buffer), dev_buffer_size * sizeof(uint32_t), stream));
#else
    CHECK_CUDA(::cudaMalloc(&(dev_buffer), dev_buffer_size * sizeof(uint32_t)));
#endif
  }
};

template <>
struct CURANDGeneratorBuilder<VariantKind::GPU> {
  static CURANDGenerator* build(BitGeneratorType gentype) { return new GPUGenerator(gentype); }

  static void destroy(CURANDGenerator* cugenptr)
  {
    // wait for rand jobs and clean-up resources
    std::lock_guard<std::mutex> guard(cugenptr->lock);
    auto stream = get_cached_stream();
// TODO: use realm allocator
#if (__CUDACC_VER_MAJOR__ > 11 || ((__CUDACC_VER_MAJOR__ >= 11) && (__CUDACC_VER_MINOR__ >= 2)))
    CHECK_CUDA(::cudaFreeAsync(cugenptr->dev_buffer, stream));
#else
    CHECK_CUDA(::cudaFree(cugenptr->dev_buffer));
#endif
    CHECK_CURAND(::curandDestroyGenerator(cugenptr->gen));
  }
};

template <>
std::map<Legion::Processor, std::unique_ptr<generator_map<VariantKind::GPU>>>
  BitGeneratorImplBody<VariantKind::GPU>::m_generators = {};

template <>
std::mutex BitGeneratorImplBody<VariantKind::GPU>::lock_generators = {};

/*static*/ void BitGeneratorTask::gpu_variant(legate::TaskContext& context)
{
  bitgenerator_template<VariantKind::GPU>(context);
}

}  // namespace cunumeric
