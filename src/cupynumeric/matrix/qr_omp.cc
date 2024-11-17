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

#include "cupynumeric/matrix/qr.h"
#include "cupynumeric/matrix/qr_template.inl"
#include "cupynumeric/matrix/qr_cpu.inl"

#include <omp.h>

namespace cupynumeric {

/*static*/ void QrTask::omp_variant(TaskContext context)
{
  openblas_set_num_threads(omp_get_max_threads());
  qr_template<VariantKind::OMP>(context);
}

}  // namespace cupynumeric
