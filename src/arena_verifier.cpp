#include "tensorkiln/arena.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "arena_support.hpp"

namespace tensorkiln {
namespace {

struct ActiveAllocation final {
  std::uint32_t buffer_ordinal;
  std::uint64_t range_end;
};

[[nodiscard]] std::string buffer_label(const std::uint32_t ordinal) {
  return "#b" + std::to_string(ordinal);
}

[[nodiscard]] Result<ArenaPlan> placement_error(const ErrorCode code,
                                                std::string message) {
  return Result<ArenaPlan>::failure(
      detail::arena_error(code, std::move(message)));
}

[[nodiscard]] Result<ArenaPlan> overlap_error(
    const ArenaAllocation& previous, const ArenaAllocation& current) {
  return placement_error(
      ErrorCode::arena_live_overlap,
      "arena buffers " + buffer_label(previous.buffer_ordinal()) + " and " +
          buffer_label(current.buffer_ordinal()) +
          " overlap in bytes [" +
          std::to_string(previous.offset_bytes()) + "," +
          std::to_string(previous.offset_bytes() +
                         previous.reserved_bytes()) +
          ") and [" + std::to_string(current.offset_bytes()) + "," +
          std::to_string(current.offset_bytes() + current.reserved_bytes()) +
          ") while lifetimes [" +
          std::to_string(previous.live_begin_step()) + "," +
          std::to_string(previous.live_end_step_exclusive()) + ") and [" +
          std::to_string(current.live_begin_step()) + "," +
          std::to_string(current.live_end_step_exclusive()) +
          ") intersect");
}

}  // namespace

Result<ArenaPlan> ArenaPlacementVerifier::verify(
    const std::span<const ArenaBufferRequest> requests,
    const std::span<const ArenaPlacement> placements,
    const ArenaLimits limits) {
  auto prepared_result = detail::prepare_arena_requests(requests, limits);
  if (!prepared_result.has_value()) {
    return Result<ArenaPlan>::failure(*prepared_result.error_if());
  }
  detail::PreparedArenaRequests prepared =
      std::move(*prepared_result.value_if());

  if (placements.size() != prepared.requests.size()) {
    return placement_error(
        ErrorCode::arena_placement_count_mismatch,
        "arena plan has " + std::to_string(placements.size()) +
            " placements; expected " +
            std::to_string(prepared.requests.size()));
  }

  std::vector<std::optional<std::uint64_t>> offsets(
      prepared.requests.size());
  for (std::size_t placement_index = 0U;
       placement_index < placements.size(); ++placement_index) {
    const ArenaPlacement& placement = placements[placement_index];
    const std::size_t ordinal =
        static_cast<std::size_t>(placement.buffer_ordinal);
    if (ordinal >= prepared.requests.size()) {
      return placement_error(
          ErrorCode::arena_placement_buffer_not_found,
          "arena placement " + std::to_string(placement_index) +
              " references unknown buffer " +
              buffer_label(placement.buffer_ordinal));
    }
    if (offsets[ordinal].has_value()) {
      return placement_error(
          ErrorCode::arena_placement_duplicate,
          "arena buffer " + buffer_label(placement.buffer_ordinal) +
              " has multiple placements");
    }
    if ((placement.offset_bytes % kArenaAlignmentBytes) != 0U) {
      return placement_error(
          ErrorCode::arena_alignment_invalid,
          "arena buffer " + buffer_label(placement.buffer_ordinal) +
              " offset " + std::to_string(placement.offset_bytes) +
              " is not 64-byte aligned");
    }
    const std::uint64_t reserved_bytes =
        prepared.requests[ordinal].reserved_bytes;
    if (placement.offset_bytes >
        std::numeric_limits<std::uint64_t>::max() - reserved_bytes) {
      return placement_error(
          ErrorCode::arena_size_overflow,
          "arena buffer " + buffer_label(placement.buffer_ordinal) +
              " offset " + std::to_string(placement.offset_bytes) +
              " plus reserved size " + std::to_string(reserved_bytes) +
              " overflows uint64");
    }
    offsets[ordinal] = placement.offset_bytes;
  }

  std::vector<ArenaAllocation> allocations;
  allocations.reserve(prepared.requests.size());
  std::uint64_t workspace_bytes = 0U;
  for (std::size_t ordinal = 0U; ordinal < prepared.requests.size();
       ++ordinal) {
    if (!offsets[ordinal].has_value()) {
      return placement_error(
          ErrorCode::compiler_internal_invariant,
          "arena placement domain omitted buffer " + buffer_label(
              static_cast<std::uint32_t>(ordinal)) +
              " after count and uniqueness validation");
    }
    const detail::PreparedArenaRequest& request = prepared.requests[ordinal];
    const std::uint64_t offset = *offsets[ordinal];
    const std::uint64_t range_end = offset + request.reserved_bytes;
    workspace_bytes = std::max(workspace_bytes, range_end);
    allocations.push_back(ArenaAllocation(
        static_cast<std::uint32_t>(ordinal), offset, request.payload_bytes,
        request.reserved_bytes, request.live_begin_step,
        request.live_end_step_exclusive));
  }

  std::vector<std::size_t> start_order(allocations.size());
  std::iota(start_order.begin(), start_order.end(), 0U);
  std::sort(
      start_order.begin(), start_order.end(),
      [&allocations](const std::size_t left, const std::size_t right) {
        const ArenaAllocation& left_allocation = allocations[left];
        const ArenaAllocation& right_allocation = allocations[right];
        if (left_allocation.live_begin_step() !=
            right_allocation.live_begin_step()) {
          return left_allocation.live_begin_step() <
                 right_allocation.live_begin_step();
        }
        return left_allocation.buffer_ordinal() <
               right_allocation.buffer_ordinal();
      });

  std::vector<std::size_t> end_order = start_order;
  std::sort(
      end_order.begin(), end_order.end(),
      [&allocations](const std::size_t left, const std::size_t right) {
        const ArenaAllocation& left_allocation = allocations[left];
        const ArenaAllocation& right_allocation = allocations[right];
        if (left_allocation.live_end_step_exclusive() !=
            right_allocation.live_end_step_exclusive()) {
          return left_allocation.live_end_step_exclusive() <
                 right_allocation.live_end_step_exclusive();
        }
        return left_allocation.buffer_ordinal() <
               right_allocation.buffer_ordinal();
      });

  std::map<std::uint64_t, ActiveAllocation> active_by_offset;
  std::size_t next_end = 0U;
  std::uint64_t live_reserved_bytes = 0U;
  std::uint64_t peak_live_reserved_bytes = 0U;

  for (const std::size_t start_index : start_order) {
    const ArenaAllocation& current = allocations[start_index];
    while (next_end < end_order.size()) {
      const ArenaAllocation& expired = allocations[end_order[next_end]];
      if (expired.live_end_step_exclusive() > current.live_begin_step()) {
        break;
      }
      const auto active = active_by_offset.find(expired.offset_bytes());
      if (active == active_by_offset.end() ||
          active->second.buffer_ordinal != expired.buffer_ordinal() ||
          live_reserved_bytes < expired.reserved_bytes()) {
        return placement_error(
            ErrorCode::compiler_internal_invariant,
            "arena verifier active-set invariant failed while expiring " +
                buffer_label(expired.buffer_ordinal()));
      }
      live_reserved_bytes -= expired.reserved_bytes();
      active_by_offset.erase(active);
      ++next_end;
    }

    const std::uint64_t current_end =
        current.offset_bytes() + current.reserved_bytes();
    const auto next = active_by_offset.lower_bound(current.offset_bytes());
    if (next != active_by_offset.begin()) {
      const auto previous = std::prev(next);
      if (previous->second.range_end > current.offset_bytes()) {
        return overlap_error(
            allocations[previous->second.buffer_ordinal], current);
      }
    }
    if (next != active_by_offset.end() && current_end > next->first) {
      return overlap_error(allocations[next->second.buffer_ordinal], current);
    }
    const auto inserted = active_by_offset.emplace(
        current.offset_bytes(),
        ActiveAllocation{current.buffer_ordinal(), current_end});
    if (!inserted.second) {
      return overlap_error(
          allocations[inserted.first->second.buffer_ordinal], current);
    }
    if (live_reserved_bytes >
        std::numeric_limits<std::uint64_t>::max() -
            current.reserved_bytes()) {
      return placement_error(
          ErrorCode::arena_size_overflow,
          "arena live reserved-byte total overflows uint64 at buffer " +
              buffer_label(current.buffer_ordinal()));
    }
    live_reserved_bytes += current.reserved_bytes();
    peak_live_reserved_bytes =
        std::max(peak_live_reserved_bytes, live_reserved_bytes);
  }

  while (next_end < end_order.size()) {
    const ArenaAllocation& expired = allocations[end_order[next_end]];
    const auto active = active_by_offset.find(expired.offset_bytes());
    if (active == active_by_offset.end() ||
        active->second.buffer_ordinal != expired.buffer_ordinal() ||
        live_reserved_bytes < expired.reserved_bytes()) {
      return placement_error(
          ErrorCode::compiler_internal_invariant,
          "arena verifier active-set invariant failed while draining " +
              buffer_label(expired.buffer_ordinal()));
    }
    live_reserved_bytes -= expired.reserved_bytes();
    active_by_offset.erase(active);
    ++next_end;
  }
  if (!active_by_offset.empty() || live_reserved_bytes != 0U) {
    return placement_error(
        ErrorCode::compiler_internal_invariant,
        "arena verifier did not drain its active allocation set");
  }

  if (workspace_bytes > limits.max_workspace_bytes) {
    return placement_error(
        ErrorCode::arena_workspace_limit_exceeded,
        "arena workspace requires " + std::to_string(workspace_bytes) +
            " bytes; limit is " +
            std::to_string(limits.max_workspace_bytes));
  }
  if (workspace_bytes > static_cast<std::uint64_t>(
                            std::numeric_limits<std::size_t>::max())) {
    return placement_error(
        ErrorCode::arena_workspace_unaddressable,
        "arena workspace requires " + std::to_string(workspace_bytes) +
            " bytes; addressable size limit is " +
            std::to_string(std::numeric_limits<std::size_t>::max()));
  }

  const auto buffer_count =
      static_cast<std::uint32_t>(allocations.size());
  const ArenaPlanStats stats{
      buffer_count,
      prepared.total_payload_bytes,
      prepared.total_reserved_bytes,
      peak_live_reserved_bytes,
      workspace_bytes,
  };
  return Result<ArenaPlan>::success(
      ArenaPlan(limits, stats, std::move(allocations)));
}

}  // namespace tensorkiln
