#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "tensorkiln/arena.hpp"

namespace tensorkiln::detail {

struct PreparedArenaRequest final {
  std::uint64_t payload_bytes;
  std::uint64_t reserved_bytes;
  std::uint32_t live_begin_step;
  std::uint32_t live_end_step_exclusive;
};

struct PreparedArenaRequests final {
  std::vector<PreparedArenaRequest> requests;
  std::uint64_t total_payload_bytes;
  std::uint64_t total_reserved_bytes;
};

[[nodiscard]] Diagnostic arena_error(ErrorCode code, std::string message);

[[nodiscard]] Result<PreparedArenaRequests> prepare_arena_requests(
    std::span<const ArenaBufferRequest> requests, ArenaLimits limits);

}  // namespace tensorkiln::detail
