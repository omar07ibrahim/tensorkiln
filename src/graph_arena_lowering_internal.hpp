#pragma once

#include <cstdint>
#include <span>

#include "tensorkiln/graph_arena.hpp"

namespace tensorkiln::detail {

[[nodiscard]] Result<GraphArenaLoweringResult>
verify_graph_arena_lowering_agreement(
    const VerifiedGraph& source,
    std::span<const ValueId> forward_values,
    std::span<const ArenaBufferRequest> forward_requests,
    std::uint32_t execution_step_count,
    const ArenaPlan& forward_plan, ArenaLimits limits);

}  // namespace tensorkiln::detail
