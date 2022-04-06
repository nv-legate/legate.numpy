/* Copyright 2022 NVIDIA Corporation
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

#include "cunumeric/pitches.h"

namespace cunumeric {

using namespace Legion;
using namespace legate;

template <VariantKind KIND, LegateTypeCode CODE, int32_t DIM>
struct SortImplBody;

static int get_rank(Domain domain, DomainPoint index_point)
{
  int domain_index = 0;
  for (int i = 0; i < domain.get_dim(); ++i) {
    if (i > 0) domain_index *= domain.hi()[i] - domain.lo()[i] + 1;
    domain_index += index_point[i];
  }
  return domain_index;
}

template <VariantKind KIND>
struct SortImpl {
  template <LegateTypeCode CODE, int32_t DIM>
  void operator()(SortArgs& args, std::vector<comm::Communicator>& comms) const
  {
    using VAL = legate_type_of<CODE>;

    auto rect = args.input.shape<DIM>();

    Pitches<DIM - 1> pitches;
    size_t volume = pitches.flatten(rect);

    int64_t segment_size  = rect.hi[DIM - 1] - rect.lo[DIM - 1] + 1;
    size_t segment_size_l = segment_size > 0 ? segment_size : 0;

    /*
     * Assumptions:
     * 1. Sort is always requested for the 'last' dimension within rect
     * 2. We have product_of_all_other_dimensions independent sort ranges
     * 3. if we have more than one participants:
     *  a) 1D-case: we perform parallel sort (via sampling)
     *  b) ND-case:
     *     * sort dimension complete -> data parallel sort
     *     * we perform parallel sort (via sampling per segment)
     */

    // we shall not return on empty rectangle in case of distributed data
    // as the process might still participate in the parallel communication
    if (((DIM > 1 && segment_size_l == args.segment_size_g) || !args.is_index_space) &&
        rect.empty())
      return;

    SortImplBody<KIND, CODE, DIM>()(args.input,
                                    args.output,
                                    pitches,
                                    rect,
                                    volume,
                                    segment_size_l,
                                    args.segment_size_g,
                                    args.argsort,
                                    args.stable,
                                    args.is_index_space,
                                    args.local_rank,
                                    args.num_ranks,
                                    args.num_sort_ranks,
                                    comms);
  }
};

template <VariantKind KIND>
static void sort_template(TaskContext& context)
{
  auto shape_span       = context.scalars()[1].values<int64_t>();
  size_t segment_size_g = shape_span[shape_span.size() - 1];
  auto domain           = context.get_launch_domain();
  size_t local_rank     = get_rank(domain, context.get_task_index());
  size_t num_ranks      = domain.get_volume();
  size_t num_sort_ranks = domain.hi()[domain.get_dim() - 1] - domain.lo()[domain.get_dim() - 1] + 1;

  SortArgs args{context.inputs()[0],
                context.outputs()[0],
                context.scalars()[0].value<bool>(),  // argsort
                context.scalars()[2].value<bool>(),  // stable
                segment_size_g,
                !context.is_single_task(),
                local_rank,
                num_ranks,
                num_sort_ranks};
  double_dispatch(
    args.input.dim(), args.input.code(), SortImpl<KIND>{}, args, context.communicators());
}

}  // namespace cunumeric
