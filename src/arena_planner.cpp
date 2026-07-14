#include "tensorkiln/arena.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <queue>
#include <set>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "arena_support.hpp"

namespace tensorkiln {
namespace {

struct ActiveBlock final {
  std::uint32_t live_end_step_exclusive;
  std::uint32_t buffer_ordinal;
  std::uint64_t offset_bytes;
  std::uint64_t reserved_bytes;
};

class ActiveBlockLater final {
 public:
  [[nodiscard]] bool operator()(const ActiveBlock& left,
                                const ActiveBlock& right) const noexcept {
    if (left.live_end_step_exclusive !=
        right.live_end_step_exclusive) {
      return left.live_end_step_exclusive >
             right.live_end_step_exclusive;
    }
    if (left.offset_bytes != right.offset_bytes) {
      return left.offset_bytes > right.offset_bytes;
    }
    return left.buffer_ordinal > right.buffer_ordinal;
  }
};

[[nodiscard]] std::string buffer_label(const std::uint32_t ordinal) {
  return "#b" + std::to_string(ordinal);
}

[[nodiscard]] Diagnostic planner_invariant(std::string message) {
  return detail::arena_error(ErrorCode::compiler_internal_invariant,
                             std::move(message));
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

class FreeBlockIndex final {
 public:
  [[nodiscard]] std::optional<Diagnostic> release(
      const std::uint64_t offset_bytes,
      const std::uint64_t reserved_bytes) {
    if (reserved_bytes == 0U ||
        (offset_bytes % kArenaAlignmentBytes) != 0U ||
        (reserved_bytes % kArenaAlignmentBytes) != 0U) {
      return planner_invariant(
          "arena planner attempted to release an invalid free block");
    }

    std::uint64_t merged_offset = offset_bytes;
    std::uint64_t merged_size = reserved_bytes;
    auto next = by_offset_.lower_bound(merged_offset);
    if (next != by_offset_.begin()) {
      const auto previous = std::prev(next);
      std::uint64_t previous_end = 0U;
      if (!checked_add(previous->first, previous->second, previous_end)) {
        return planner_invariant(
            "arena planner free-block end overflowed uint64");
      }
      if (previous_end > merged_offset) {
        return planner_invariant(
            "arena planner released a block overlapping free storage");
      }
      if (previous_end == merged_offset) {
        if (!checked_add(previous->second, merged_size, merged_size)) {
          return planner_invariant(
              "arena planner free-block coalescing overflowed uint64");
        }
        merged_offset = previous->first;
        if (const auto error = erase(previous)) {
          return error;
        }
      }
    }

    next = by_offset_.lower_bound(merged_offset);
    if (next != by_offset_.end()) {
      std::uint64_t merged_end = 0U;
      if (!checked_add(merged_offset, merged_size, merged_end)) {
        return planner_invariant(
            "arena planner free-block end overflowed uint64");
      }
      if (merged_end > next->first) {
        return planner_invariant(
            "arena planner released a block overlapping free storage");
      }
      if (merged_end == next->first) {
        if (!checked_add(merged_size, next->second, merged_size)) {
          return planner_invariant(
              "arena planner free-block coalescing overflowed uint64");
        }
        if (const auto error = erase(next)) {
          return error;
        }
      }
    }

    const auto offset_inserted =
        by_offset_.emplace(merged_offset, merged_size);
    if (!offset_inserted.second) {
      return planner_invariant(
          "arena planner duplicated a free-block offset");
    }
    const auto size_inserted =
        by_size_.emplace(merged_size, merged_offset);
    if (!size_inserted.second) {
      by_offset_.erase(offset_inserted.first);
      return planner_invariant(
          "arena planner duplicated a free-block size index entry");
    }
    return std::nullopt;
  }

  [[nodiscard]] std::optional<Diagnostic> take_best_fit(
      const std::uint64_t required_bytes,
      std::optional<std::uint64_t>& selected_offset) {
    selected_offset.reset();
    const auto best = by_size_.lower_bound({required_bytes, 0U});
    if (best == by_size_.end()) {
      return std::nullopt;
    }

    const std::uint64_t block_size = best->first;
    const std::uint64_t block_offset = best->second;
    const auto by_offset = by_offset_.find(block_offset);
    if (by_offset == by_offset_.end() || by_offset->second != block_size) {
      return planner_invariant(
          "arena planner free-block indices disagree during allocation");
    }
    by_size_.erase(best);
    by_offset_.erase(by_offset);

    if (block_size > required_bytes) {
      std::uint64_t tail_offset = 0U;
      if (!checked_add(block_offset, required_bytes, tail_offset)) {
        return planner_invariant(
            "arena planner split offset overflowed uint64");
      }
      if (const auto error =
              release(tail_offset, block_size - required_bytes)) {
        return error;
      }
    }
    selected_offset = block_offset;
    return std::nullopt;
  }

  [[nodiscard]] std::optional<Diagnostic> take_frontier_block(
      const std::uint64_t high_water_mark,
      std::optional<std::pair<std::uint64_t, std::uint64_t>>& block) {
    block.reset();
    if (by_offset_.empty()) {
      return std::nullopt;
    }
    const auto frontier = std::prev(by_offset_.end());
    std::uint64_t frontier_end = 0U;
    if (!checked_add(frontier->first, frontier->second, frontier_end)) {
      return planner_invariant(
          "arena planner frontier block end overflowed uint64");
    }
    if (frontier_end != high_water_mark) {
      return std::nullopt;
    }
    block = std::pair<std::uint64_t, std::uint64_t>{
        frontier->first,
        frontier->second,
    };
    return erase(frontier);
  }

 private:
  using OffsetIterator = std::map<std::uint64_t, std::uint64_t>::iterator;

  [[nodiscard]] std::optional<Diagnostic> erase(
      const OffsetIterator block) {
    const auto size_entry = by_size_.find({block->second, block->first});
    if (size_entry == by_size_.end()) {
      return planner_invariant(
          "arena planner free-block indices disagree during removal");
    }
    by_size_.erase(size_entry);
    by_offset_.erase(block);
    return std::nullopt;
  }

  std::map<std::uint64_t, std::uint64_t> by_offset_;
  std::set<std::pair<std::uint64_t, std::uint64_t>> by_size_;
};

}  // namespace

Result<ArenaPlan> ArenaPlanner::run(
    const std::span<const ArenaBufferRequest> requests,
    const ArenaLimits limits) {
  auto prepared_result = detail::prepare_arena_requests(requests, limits);
  if (!prepared_result.has_value()) {
    return Result<ArenaPlan>::failure(*prepared_result.error_if());
  }
  detail::PreparedArenaRequests prepared =
      std::move(*prepared_result.value_if());

  std::vector<std::size_t> allocation_order(prepared.requests.size());
  for (std::size_t index = 0U; index < allocation_order.size(); ++index) {
    allocation_order[index] = index;
  }
  std::sort(
      allocation_order.begin(), allocation_order.end(),
      [&prepared](const std::size_t left, const std::size_t right) {
        const detail::PreparedArenaRequest& left_request =
            prepared.requests[left];
        const detail::PreparedArenaRequest& right_request =
            prepared.requests[right];
        if (left_request.live_begin_step != right_request.live_begin_step) {
          return left_request.live_begin_step < right_request.live_begin_step;
        }
        return left < right;
      });

  std::priority_queue<ActiveBlock, std::vector<ActiveBlock>,
                      ActiveBlockLater>
      active;
  FreeBlockIndex free_blocks;
  std::vector<ArenaPlacement> placements;
  placements.reserve(prepared.requests.size());
  std::uint64_t high_water_mark = 0U;

  for (const std::size_t index : allocation_order) {
    const auto ordinal = static_cast<std::uint32_t>(index);
    const detail::PreparedArenaRequest& request = prepared.requests[index];
    while (!active.empty() &&
           active.top().live_end_step_exclusive <=
               request.live_begin_step) {
      const ActiveBlock expired = active.top();
      active.pop();
      if (const auto error = free_blocks.release(
              expired.offset_bytes, expired.reserved_bytes)) {
        return Result<ArenaPlan>::failure(*error);
      }
    }

    std::optional<std::uint64_t> selected_offset;
    if (const auto error =
            free_blocks.take_best_fit(request.reserved_bytes,
                                      selected_offset)) {
      return Result<ArenaPlan>::failure(*error);
    }
    std::uint64_t offset = 0U;
    if (selected_offset.has_value()) {
      offset = *selected_offset;
    } else {
      std::optional<std::pair<std::uint64_t, std::uint64_t>> frontier;
      if (const auto error =
              free_blocks.take_frontier_block(high_water_mark, frontier)) {
        return Result<ArenaPlan>::failure(*error);
      }
      std::uint64_t extension_bytes = request.reserved_bytes;
      if (frontier.has_value()) {
        offset = frontier->first;
        if (frontier->second >= request.reserved_bytes) {
          return Result<ArenaPlan>::failure(planner_invariant(
              "arena planner bypassed a fitting frontier block"));
        }
        extension_bytes -= frontier->second;
      } else {
        offset = high_water_mark;
      }
      if (!checked_add(high_water_mark, extension_bytes, high_water_mark)) {
        return Result<ArenaPlan>::failure(detail::arena_error(
            ErrorCode::arena_size_overflow,
            "arena planner workspace overflows uint64 at buffer " +
                buffer_label(ordinal)));
      }
    }

    placements.push_back(ArenaPlacement{ordinal, offset});
    active.push(ActiveBlock{
        request.live_end_step_exclusive,
        ordinal,
        offset,
        request.reserved_bytes,
    });
  }

  auto verified = ArenaPlacementVerifier::verify(requests, placements, limits);
  if (verified.has_value()) {
    return verified;
  }
  if (verified.error_if()->code ==
          ErrorCode::arena_workspace_limit_exceeded ||
      verified.error_if()->code == ErrorCode::arena_workspace_unaddressable) {
    return verified;
  }
  return Result<ArenaPlan>::failure(planner_invariant(
      "arena planner produced an invalid placement: " +
      std::string(error_code_name(verified.error_if()->code)) + ": " +
      verified.error_if()->message));
}

}  // namespace tensorkiln
