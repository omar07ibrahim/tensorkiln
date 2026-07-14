#include "tensorkiln/graph_arena.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace tensorkiln {

GraphArenaLoweringResult::GraphArenaLoweringResult(
    const std::uint32_t source_node_count,
    const std::uint32_t execution_step_count,
    std::vector<ValueId> values_by_buffer_ordinal,
    std::vector<ArenaBufferRequest> requests, ArenaPlan arena_plan,
    std::vector<std::uint32_t> buffer_by_source_ordinal)
    : source_node_count_(source_node_count),
      execution_step_count_(execution_step_count),
      values_by_buffer_ordinal_(std::move(values_by_buffer_ordinal)),
      requests_(std::move(requests)),
      arena_plan_(std::move(arena_plan)),
      buffer_by_source_ordinal_(std::move(buffer_by_source_ordinal)) {}

std::optional<std::uint32_t> GraphArenaLoweringResult::buffer_ordinal(
    const ValueId value) const noexcept {
  const std::size_t source_ordinal =
      static_cast<std::size_t>(value.ordinal());
  if (source_ordinal >= buffer_by_source_ordinal_.size()) {
    return std::nullopt;
  }
  const std::uint32_t buffer_ordinal =
      buffer_by_source_ordinal_[source_ordinal];
  if (buffer_ordinal == std::numeric_limits<std::uint32_t>::max()) {
    return std::nullopt;
  }
  const std::size_t buffer_index =
      static_cast<std::size_t>(buffer_ordinal);
  if (buffer_index >= values_by_buffer_ordinal_.size() ||
      values_by_buffer_ordinal_[buffer_index] != value) {
    return std::nullopt;
  }
  return buffer_ordinal;
}

const ValueId* GraphArenaLoweringResult::value_at(
    const std::uint32_t buffer_ordinal) const noexcept {
  const std::size_t index = static_cast<std::size_t>(buffer_ordinal);
  if (index >= values_by_buffer_ordinal_.size()) {
    return nullptr;
  }
  return &values_by_buffer_ordinal_[index];
}

const ArenaAllocation* GraphArenaLoweringResult::allocation_for(
    const ValueId value) const noexcept {
  const std::optional<std::uint32_t> ordinal = buffer_ordinal(value);
  if (!ordinal.has_value()) {
    return nullptr;
  }
  return arena_plan_.allocation_at(*ordinal);
}

std::string GraphArenaLoweringResult::dump() const {
  std::string result{"tensorkiln.graph_arena_lowering v0 {\n"};
  result += "  source_nodes=" + std::to_string(source_node_count_) + "\n";
  result += "  execution_steps=" +
            std::to_string(execution_step_count_) + "\n";
  result += "  buffers=" +
            std::to_string(values_by_buffer_ordinal_.size()) + "\n";
  for (std::size_t index = 0U;
       index < values_by_buffer_ordinal_.size(); ++index) {
    const ValueId value = values_by_buffer_ordinal_[index];
    const ArenaBufferRequest& request = requests_[index];
    result += "  #b" + std::to_string(index) + " <- #n" +
              std::to_string(value.ordinal()) + " %" +
              std::to_string(value.ordinal()) + " step=" +
              std::to_string(request.live_begin_step) + "\n";
  }
  result += "}\n";
  result += arena_plan_.dump();
  return result;
}

}  // namespace tensorkiln
