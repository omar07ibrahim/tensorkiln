#include "tensorkiln/arena.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace tensorkiln {

ArenaAllocation::ArenaAllocation(const std::uint32_t buffer_ordinal,
                                 const std::uint64_t offset_bytes,
                                 const std::uint64_t payload_bytes,
                                 const std::uint64_t reserved_bytes,
                                 const std::uint32_t live_begin_step,
                                 const std::uint32_t
                                     live_end_step_exclusive) noexcept
    : buffer_ordinal_(buffer_ordinal),
      offset_bytes_(offset_bytes),
      payload_bytes_(payload_bytes),
      reserved_bytes_(reserved_bytes),
      live_begin_step_(live_begin_step),
      live_end_step_exclusive_(live_end_step_exclusive) {}

ArenaPlan::ArenaPlan(const ArenaLimits limits, const ArenaPlanStats stats,
                     std::vector<ArenaAllocation> allocations)
    : limits_(limits), stats_(stats), allocations_(std::move(allocations)) {}

const ArenaAllocation* ArenaPlan::allocation_at(
    const std::uint32_t buffer_ordinal) const noexcept {
  const std::size_t index = static_cast<std::size_t>(buffer_ordinal);
  if (index >= allocations_.size() ||
      allocations_[index].buffer_ordinal() != buffer_ordinal) {
    return nullptr;
  }
  return &allocations_[index];
}

std::string ArenaPlan::dump() const {
  std::string result{"tensorkiln.arena_plan v0 {\n"};
  result += "  alignment_bytes=" + std::to_string(kArenaAlignmentBytes) +
            "\n";
  result += "  limits {buffers=" + std::to_string(limits_.max_buffers) +
            ", workspace_bytes=" +
            std::to_string(limits_.max_workspace_bytes) + "}\n";
  result += "  stats {buffers=" + std::to_string(stats_.buffer_count) +
            ", payload_bytes=" +
            std::to_string(stats_.total_payload_bytes) +
            ", reserved_bytes=" +
            std::to_string(stats_.total_reserved_bytes) +
            ", peak_live_reserved_bytes=" +
            std::to_string(stats_.peak_live_reserved_bytes) +
            ", workspace_bytes=" +
            std::to_string(stats_.workspace_bytes) + "}\n";
  for (const ArenaAllocation& allocation : allocations_) {
    result += "  #b" + std::to_string(allocation.buffer_ordinal()) +
              " offset=" + std::to_string(allocation.offset_bytes()) +
              " payload=" + std::to_string(allocation.payload_bytes()) +
              " reserved=" + std::to_string(allocation.reserved_bytes()) +
              " live=[" + std::to_string(allocation.live_begin_step()) +
              "," +
              std::to_string(allocation.live_end_step_exclusive()) + ")\n";
  }
  result += "}\n";
  return result;
}

}  // namespace tensorkiln
