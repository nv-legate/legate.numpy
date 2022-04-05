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

#include "cunumeric/scan/cumsum_g.h"
#include "cunumeric/scan/cumsum_g_template.inl"

#include <thrust/scan.h>
#include <thrust/execution_policy.h>


namespace cunumeric {

using namespace Legion;
using namespace legate;

template <LegateTypeCode CODE, int DIM>
struct Cumsum_gImplBody<VariantKind::CPU, CODE, DIM> {
  using VAL = legate_type_of<CODE>;

  size_t operator()(const AccessorWO<VAL, DIM>& out,
		    const AccessorRO<VAL, DIM>& in,
		    const AccessorWO<VAL, DIM>& sum_vals,
                    const Pitches<DIM - 1>& pitches,
                    const Rect<DIM>& rect,
		    const int axis,
		    const DomainPoint& partition_index)

  {
    
  }
};

/*static*/ void Cumsum_gTask::cpu_variant(TaskContext& context)
{
  cumsum_g_template<VariantKind::CPU>(context);
}

namespace  // unnamed
{
static void __attribute__((constructor)) register_tasks(void) { NonzeroTask::register_variants(); }
}  // namespace

}  // namespace cunumeric
  
