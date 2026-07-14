#include "tensorkiln/graph_arena.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace tensorkiln {
namespace {

class MaterializesArenaBuffer final {
 public:
  [[nodiscard]] bool operator()(const InputOp&) const noexcept {
    return false;
  }
  [[nodiscard]] bool operator()(const ConstantOp&) const noexcept {
    return false;
  }
  [[nodiscard]] bool operator()(const AddOp&) const noexcept { return true; }
  [[nodiscard]] bool operator()(const MatMulOp&) const noexcept {
    return true;
  }
  [[nodiscard]] bool operator()(const ReluOp&) const noexcept { return true; }
};

[[nodiscard]] bool materializes_arena_buffer(
    const Operation& operation) noexcept {
  return std::visit(MaterializesArenaBuffer{}, operation);
}

[[nodiscard]] Result<GraphArenaLoweringResult> invariant_error(
    std::string message) {
  return Result<GraphArenaLoweringResult>::failure(Diagnostic{
      ErrorCode::compiler_internal_invariant,
      std::move(message),
  });
}

}  // namespace

Result<GraphArenaLoweringResult> GraphArenaPlacementVerifier::verify(
    const VerifiedGraph& source,
    const std::span<const ArenaPlacement> placements,
    const ArenaLimits limits) {
  const std::span<const Node> nodes = source.nodes();
  if (nodes.size() > static_cast<std::size_t>(
                         std::numeric_limits<std::uint32_t>::max())) {
    return invariant_error(
        "graph arena source node count exceeds the uint32 domain");
  }

  std::size_t materialized_buffer_count = 0U;
  for (std::size_t source_index = 0U; source_index < nodes.size();
       ++source_index) {
    const Node& node = nodes[source_index];
    if (node.id().ordinal() != source_index ||
        node.output().ordinal() != source_index ||
        source.node(node.id()) != &node ||
        source.type(node.output()) != &node.output_type()) {
      return invariant_error(
          "graph arena source has a non-dense or foreign definition at index " +
          std::to_string(source_index));
    }
    if (materializes_arena_buffer(node.operation())) {
      ++materialized_buffer_count;
    }
  }
  if (materialized_buffer_count >
      static_cast<std::size_t>(limits.max_buffers)) {
    return Result<GraphArenaLoweringResult>::failure(Diagnostic{
        ErrorCode::arena_buffer_limit_exceeded,
        "arena has " + std::to_string(materialized_buffer_count) +
            " buffer requests; limit is " +
            std::to_string(limits.max_buffers),
    });
  }

  std::vector<std::optional<std::uint32_t>> buffer_by_source_ordinal(
      nodes.size());
  std::vector<ValueId> values_by_buffer_ordinal;
  std::vector<std::uint64_t> payload_bytes;
  std::vector<std::uint32_t> live_begin_steps;
  std::vector<std::uint32_t> live_end_steps;
  values_by_buffer_ordinal.reserve(materialized_buffer_count);
  payload_bytes.reserve(materialized_buffer_count);
  live_begin_steps.reserve(materialized_buffer_count);
  live_end_steps.reserve(materialized_buffer_count);

  std::uint64_t next_step = 0U;
  for (std::size_t source_index = 0U; source_index < nodes.size();
       ++source_index) {
    const Node& node = nodes[source_index];
    if (!materializes_arena_buffer(node.operation())) {
      continue;
    }
    if (next_step >=
        static_cast<std::uint64_t>(
            std::numeric_limits<std::uint32_t>::max())) {
      return invariant_error(
          "graph arena execution step count exceeds the uint32 domain");
    }
    if (values_by_buffer_ordinal.size() >=
        static_cast<std::size_t>(
            std::numeric_limits<std::uint32_t>::max())) {
      return invariant_error(
          "graph arena buffer count exceeds the uint32 domain");
    }

    const auto step = static_cast<std::uint32_t>(next_step);
    const auto buffer_ordinal = static_cast<std::uint32_t>(
        values_by_buffer_ordinal.size());
    buffer_by_source_ordinal[source_index] = buffer_ordinal;
    values_by_buffer_ordinal.push_back(node.output());

    const std::size_t node_payload_bytes = node.output_type().byte_count();
    if constexpr (sizeof(std::size_t) > sizeof(std::uint64_t)) {
      if (node_payload_bytes > static_cast<std::size_t>(
                                   std::numeric_limits<std::uint64_t>::max())) {
        return invariant_error(
            "graph arena tensor byte count exceeds the uint64 domain at #n" +
            std::to_string(node.id().ordinal()));
      }
    }
    payload_bytes.push_back(
        static_cast<std::uint64_t>(node_payload_bytes));
    live_begin_steps.push_back(step);
    live_end_steps.push_back(step + 1U);
    ++next_step;
  }

  const auto execution_step_count =
      static_cast<std::uint32_t>(next_step);
  for (std::size_t output_index = 0U;
       output_index < source.outputs().size(); ++output_index) {
    const GraphOutput& output = source.outputs()[output_index];
    const std::size_t source_ordinal =
        static_cast<std::size_t>(output.value().ordinal());
    if (output.id().ordinal != output_index ||
        source_ordinal >= nodes.size() ||
        source.type(output.value()) == nullptr ||
        nodes[source_ordinal].output() != output.value()) {
      return invariant_error(
          "graph arena source has an invalid output at index " +
          std::to_string(output_index));
    }
    const std::optional<std::uint32_t> buffer_ordinal =
        buffer_by_source_ordinal[source_ordinal];
    if (buffer_ordinal.has_value()) {
      live_end_steps[static_cast<std::size_t>(*buffer_ordinal)] =
          execution_step_count;
    } else if (materializes_arena_buffer(
                   nodes[source_ordinal].operation())) {
      return invariant_error(
          "graph arena output mapping omitted a materialized value");
    }
  }

  std::uint64_t reverse_step = next_step;
  for (auto node_iterator = nodes.rbegin(); node_iterator != nodes.rend();
       ++node_iterator) {
    const Node& node = *node_iterator;
    if (!materializes_arena_buffer(node.operation())) {
      continue;
    }
    if (reverse_step == 0U) {
      return invariant_error(
          "graph arena reverse execution step underflowed");
    }
    --reverse_step;
    const std::size_t source_index =
        static_cast<std::size_t>(node.output().ordinal());
    const std::optional<std::uint32_t> current_buffer =
        buffer_by_source_ordinal[source_index];
    if (!current_buffer.has_value() ||
        *current_buffer != static_cast<std::uint32_t>(reverse_step)) {
      return invariant_error(
          "graph arena reverse execution order is not dense at #n" +
          std::to_string(node.id().ordinal()));
    }
    const auto consumer_boundary =
        static_cast<std::uint32_t>(reverse_step + 1U);

    for (const ValueId operand : node.inputs()) {
      const std::size_t operand_ordinal =
          static_cast<std::size_t>(operand.ordinal());
      if (operand_ordinal >= source_index || operand_ordinal >= nodes.size() ||
          source.type(operand) == nullptr ||
          nodes[operand_ordinal].output() != operand) {
        return invariant_error(
            "graph arena source has a foreign or forward operand %" +
            std::to_string(operand.ordinal()) + " at #n" +
            std::to_string(node.id().ordinal()));
      }
      const std::optional<std::uint32_t> operand_buffer =
          buffer_by_source_ordinal[operand_ordinal];
      if (operand_buffer.has_value()) {
        const std::size_t buffer_index =
            static_cast<std::size_t>(*operand_buffer);
        live_end_steps[buffer_index] =
            std::max(live_end_steps[buffer_index], consumer_boundary);
      } else if (materializes_arena_buffer(
                     nodes[operand_ordinal].operation())) {
        return invariant_error(
            "graph arena operand mapping omitted a materialized value");
      }
    }
  }
  if (reverse_step != 0U) {
    return invariant_error(
        "graph arena reverse execution steps did not drain");
  }

  std::vector<ArenaBufferRequest> requests;
  requests.reserve(values_by_buffer_ordinal.size());
  for (std::size_t buffer_index = 0U;
       buffer_index < values_by_buffer_ordinal.size(); ++buffer_index) {
    requests.push_back(ArenaBufferRequest{
        payload_bytes[buffer_index],
        live_begin_steps[buffer_index],
        live_end_steps[buffer_index],
    });
  }

  auto verified = ArenaPlacementVerifier::verify(requests, placements, limits);
  if (!verified.has_value()) {
    return Result<GraphArenaLoweringResult>::failure(*verified.error_if());
  }
  ArenaPlan arena_plan = std::move(*verified.value_if());

  constexpr std::uint32_t missing_buffer =
      std::numeric_limits<std::uint32_t>::max();
  std::vector<std::uint32_t> dense_buffer_by_source_ordinal(
      nodes.size(), missing_buffer);
  for (std::size_t source_index = 0U; source_index < nodes.size();
       ++source_index) {
    if (buffer_by_source_ordinal[source_index].has_value()) {
      dense_buffer_by_source_ordinal[source_index] =
          *buffer_by_source_ordinal[source_index];
    }
  }

  return Result<GraphArenaLoweringResult>::success(
      GraphArenaLoweringResult(
          static_cast<std::uint32_t>(nodes.size()), execution_step_count,
          std::move(values_by_buffer_ordinal), std::move(requests),
          std::move(arena_plan),
          std::move(dense_buffer_by_source_ordinal)));
}

}  // namespace tensorkiln
