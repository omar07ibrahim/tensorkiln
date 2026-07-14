#include "arena_support.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace tensorkiln::detail {
namespace {

[[nodiscard]] std::string buffer_label(const std::size_t ordinal) {
  return "#b" + std::to_string(ordinal);
}

[[nodiscard]] bool checked_add(const std::uint64_t left,
                               const std::uint64_t right,
                               std::uint64_t& result) noexcept {
  if (left > std::numeric_limits<std::uint64_t>::max() - right) {
    return false;
  }
  result = left + right;
  return true;
}

}  // namespace

Diagnostic arena_error(const ErrorCode code, std::string message) {
  return Diagnostic{code, std::move(message)};
}

Result<PreparedArenaRequests> prepare_arena_requests(
    const std::span<const ArenaBufferRequest> requests,
    const ArenaLimits limits) {
  if (requests.size() >
      static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()) ||
      requests.size() > static_cast<std::size_t>(limits.max_buffers)) {
    return Result<PreparedArenaRequests>::failure(arena_error(
        ErrorCode::arena_buffer_limit_exceeded,
        "arena has " + std::to_string(requests.size()) +
            " buffer requests; limit is " +
            std::to_string(limits.max_buffers)));
  }

  std::vector<PreparedArenaRequest> prepared;
  prepared.reserve(requests.size());
  std::uint64_t total_payload_bytes = 0U;
  std::uint64_t total_reserved_bytes = 0U;

  for (std::size_t ordinal = 0U; ordinal < requests.size(); ++ordinal) {
    const ArenaBufferRequest& request = requests[ordinal];
    if (request.size_bytes == 0U) {
      return Result<PreparedArenaRequests>::failure(arena_error(
          ErrorCode::arena_buffer_size_invalid,
          "arena buffer " + buffer_label(ordinal) +
              " has zero payload bytes"));
    }
    if (request.live_begin_step >= request.live_end_step_exclusive) {
      return Result<PreparedArenaRequests>::failure(arena_error(
          ErrorCode::arena_lifetime_invalid,
          "arena buffer " + buffer_label(ordinal) +
              " has invalid lifetime [" +
              std::to_string(request.live_begin_step) + "," +
              std::to_string(request.live_end_step_exclusive) +
              "); begin must be less than end"));
    }
    if (request.size_bytes >
        std::numeric_limits<std::uint64_t>::max() -
            (kArenaAlignmentBytes - 1U)) {
      return Result<PreparedArenaRequests>::failure(arena_error(
          ErrorCode::arena_size_overflow,
          "arena buffer " + buffer_label(ordinal) + " payload size " +
              std::to_string(request.size_bytes) +
              " cannot be rounded to 64-byte alignment"));
    }
    const std::uint64_t reserved_bytes =
        (request.size_bytes + (kArenaAlignmentBytes - 1U)) &
        ~(kArenaAlignmentBytes - 1U);

    std::uint64_t next_total = 0U;
    if (!checked_add(total_payload_bytes, request.size_bytes, next_total)) {
      return Result<PreparedArenaRequests>::failure(arena_error(
          ErrorCode::arena_size_overflow,
          "arena payload total overflows uint64 at buffer " +
              buffer_label(ordinal)));
    }
    total_payload_bytes = next_total;
    if (!checked_add(total_reserved_bytes, reserved_bytes, next_total)) {
      return Result<PreparedArenaRequests>::failure(arena_error(
          ErrorCode::arena_size_overflow,
          "arena reserved-byte total overflows uint64 at buffer " +
              buffer_label(ordinal)));
    }
    total_reserved_bytes = next_total;
    prepared.push_back(PreparedArenaRequest{
        request.size_bytes,
        reserved_bytes,
        request.live_begin_step,
        request.live_end_step_exclusive,
    });
  }

  return Result<PreparedArenaRequests>::success(PreparedArenaRequests{
      std::move(prepared),
      total_payload_bytes,
      total_reserved_bytes,
  });
}

}  // namespace tensorkiln::detail
