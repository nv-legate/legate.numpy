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

#pragma once

#include <unistd.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <mutex>

#include "cunumeric/random/bitgenerator.h"
#include "cunumeric/random/bitgenerator_template.inl"
#include "cunumeric/random/bitgenerator_util.h"

#include "cunumeric/random/curand_help.h"
#include "cunumeric/random/curandex/curand_ex.h"

namespace cunumeric {

using namespace Legion;
using namespace legate;

static Logger log_curand("cunumeric.random");

template <VariantKind kind>
struct CURANDGeneratorBuilder;

struct CURANDGenerator {
  curandGeneratorEx_t gen;
  uint64_t seed;
  uint64_t generatorId;
  curandRngType type;

 protected:
  CURANDGenerator(BitGeneratorType gentype, uint64_t seed, uint64_t generatorId)
    : type(get_curandRngType(gentype)), seed(seed), generatorId(generatorId)
  {
    log_curand.debug() << "CURANDGenerator::create";
  }

  CURANDGenerator(const CURANDGenerator&) = delete;

 public:
  virtual ~CURANDGenerator() { log_curand.debug() << "CURANDGenerator::destroy"; }

  void generate_raw(uint64_t count, uint32_t* out)
  {
    CHECK_CURAND(::curandGenerateRawUInt32Ex(gen, out, count));
  }
};

struct generate_fn {
  template <int32_t DIM>
  size_t operator()(CURANDGenerator& gen, legate::Store& output)
  {
    auto rect       = output.shape<DIM>();
    uint64_t volume = rect.volume();

    const auto proc = Legion::Processor::get_executing_processor();
    log_curand.debug() << "proc=" << proc << " - shape = " << rect;

    if (volume > 0) {
      auto out = output.write_accessor<uint32_t, DIM>(rect);

      uint32_t* p = out.ptr(rect);

      gen.generate_raw(volume, p);
    }

    return volume;
  }
};

template <VariantKind kind>
struct generator_map {
  generator_map() {}
  ~generator_map()
  {
    std::lock_guard<std::mutex> guard(lock);
    if (m_generators.size() != 0) {
      log_curand.debug() << "some generators have not been freed - cleaning-up !";
      // actually destroy
      for (auto kv = m_generators.begin(); kv != m_generators.end(); ++kv) {
        auto cugenptr = kv->second;
        CURANDGeneratorBuilder<kind>::destroy(cugenptr);
      }
      m_generators.clear();
    }
  }

  std::mutex lock;
  std::map<uint32_t, CURANDGenerator*> m_generators;

  bool has(uint32_t generatorID)
  {
    std::lock_guard<std::mutex> guard(lock);
    return m_generators.find(generatorID) != m_generators.end();
  }

  CURANDGenerator* get(uint32_t generatorID)
  {
    std::lock_guard<std::mutex> guard(lock);
    if (m_generators.find(generatorID) == m_generators.end()) {
      log_curand.fatal() << "internal error : generator ID <" << generatorID
                         << "> does not exist (get) !";
      assert(false);
    }
    return m_generators[generatorID];
  }

  // called by the processor later using the generator
  void create(uint32_t generatorID, BitGeneratorType gentype, uint64_t seed, uint32_t flags)
  {
    const auto proc = Legion::Processor::get_executing_processor();
    CURANDGenerator* cugenptr =
      CURANDGeneratorBuilder<kind>::build(gentype, seed, (uint64_t)proc.id, flags);

    std::lock_guard<std::mutex> guard(lock);
    // safety check
    if (m_generators.find(generatorID) != m_generators.end()) {
      log_curand.fatal() << "internal error : generator ID <" << generatorID
                         << "> already in use !";
      assert(false);
    }
    m_generators[generatorID] = cugenptr;
  }

  void destroy(uint32_t generatorID)
  {
    CURANDGenerator* cugenptr;
    // verify it existed, and otherwise remove it from list
    {
      std::lock_guard<std::mutex> guard(lock);
      if (m_generators.find(generatorID) == m_generators.end()) {
        log_curand.fatal() << "internal error : generator ID <" << generatorID
                           << "> does not exist (destroy) !";
        assert(false);
      }
      cugenptr = m_generators[generatorID];
      m_generators.erase(generatorID);
    }

    CURANDGeneratorBuilder<kind>::destroy(cugenptr);
  }
};

template <VariantKind kind>
struct BitGeneratorImplBody {
  using generator_map_t = generator_map<kind>;

  static std::mutex lock_generators;
  static std::map<Legion::Processor, std::unique_ptr<generator_map_t>> m_generators;

 private:
  static generator_map_t& get_generator_map()
  {
    const auto proc = Legion::Processor::get_executing_processor();
    std::lock_guard<std::mutex> guard(lock_generators);
    if (m_generators.find(proc) == m_generators.end()) {
      m_generators[proc] = std::make_unique<generator_map_t>();
    }
    generator_map_t* res = m_generators[proc].get();
    return *res;
  }

 public:
  void operator()(
    BitGeneratorOperation op,
    int32_t generatorID,
    uint32_t generatorType,  // to allow for lazy initialization, generatorType is always passed
    uint64_t seed,           // to allow for lazy initialization, seed is always passed
    uint32_t flags,          // for future use - ordering, etc.
    const DomainPoint& strides,
    std::vector<legate::Store>& output,
    std::vector<legate::Store>& args)
  {
    generator_map_t& genmap = get_generator_map();
    // printtid((int)op);
    switch (op) {
      case BitGeneratorOperation::CREATE: {
        genmap.create(generatorID, static_cast<BitGeneratorType>(generatorType), seed, flags);

        log_curand.debug() << "created generator " << generatorID;
        break;
      }
      case BitGeneratorOperation::DESTROY: {
        genmap.destroy(generatorID);

        log_curand.debug() << "destroyed generator " << generatorID;
        break;
      }
      case BitGeneratorOperation::RAND_RAW: {
        // allow for lazy initialization
        if (!genmap.has(generatorID))
          genmap.create(generatorID, static_cast<BitGeneratorType>(generatorType), seed, flags);
        // get the generator
        CURANDGenerator* genptr = genmap.get(generatorID);
        if (output.size() == 0) {
          assert(false);  // TODO for skip ahead ?
        } else {
          legate::Store& res     = output[0];
          CURANDGenerator& cugen = *genptr;
          dim_dispatch(res.dim(), generate_fn{}, cugen, res);
        }
        break;
      }
      default: {
        log_curand.fatal() << "unknown BitGenerator operation";
        assert(false);
      }
    }
  }
};

}  // namespace cunumeric