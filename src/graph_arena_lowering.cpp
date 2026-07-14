#include "graph_arena_lowering_internal.hpp"

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

class ForwardMaterializesArenaBuffer final {
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

[[nodiscard]] bool forward_materializes_arena_buffer(
    const Operation& operation) noexcept {
  return std::visit(ForwardMaterializesArenaBuffer{}, operation);
}

[[nodiscard]] Result<GraphArenaLoweringResult> lowering_invariant(
    std::string message) {
  return Result<GraphArenaLoweringResult>::failure(Diagnostic{
      ErrorCode::compiler_internal_invariant,
      std::move(message),
  });
}

[[nodiscard]] Result<GraphArenaLoweringResult> planner_failure(
    const Diagnostic& error) {
  if (error.code == ErrorCode::arena_buffer_size_invalid ||
      error.code == ErrorCode::arena_lifetime_invalid) {
    return lowering_invariant(
        "graph arena lowering derived an invalid request: " +
        std::string(error_code_name(error.code)) + ": " + error.message);
  }
  if (error.code == ErrorCode::arena_buffer_limit_exceeded ||
      error.code == ErrorCode::arena_size_overflow ||
      error.code == ErrorCode::arena_workspace_limit_exceeded ||
      error.code == ErrorCode::arena_workspace_unaddressable ||
      error.code == ErrorCode::compiler_internal_invariant) {
    return Result<GraphArenaLoweringResult>::failure(error);
  }
  return lowering_invariant(
      "graph arena planner returned an unexpected diagnostic: " +
      std::string(error_code_name(error.code)) + ": " + error.message);
}

[[nodiscard]] bool same_allocation(const ArenaAllocation& left,
                                   const ArenaAllocation& right) noexcept {
  return left.buffer_ordinal() == right.buffer_ordinal() &&
         left.offset_bytes() == right.offset_bytes() &&
         left.payload_bytes() == right.payload_bytes() &&
         left.reserved_bytes() == right.reserved_bytes() &&
         left.live_begin_step() == right.live_begin_step() &&
         left.live_end_step_exclusive() ==
             right.live_end_step_exclusive();
}

}  // namespace

namespace detail {

Result<GraphArenaLoweringResult> verify_graph_arena_lowering_agreement(
    const VerifiedGraph& source,
    const std::span<const ValueId> forward_values,
    const std::span<const ArenaBufferRequest> forward_requests,
    const std::uint32_t execution_step_count,
    const ArenaPlan& forward_plan, const ArenaLimits limits) {
  const std::span<const ArenaAllocation> forward_allocations =
      forward_plan.allocations();
  if (forward_plan.limits() != limits) {
    return lowering_invariant(
        "graph arena planner changed the requested limits");
  }
  if (forward_allocations.size() != forward_requests.size()) {
    return lowering_invariant(
        "graph arena planner returned an allocation count mismatch");
  }

  std::vector<ArenaPlacement> placements;
  placements.reserve(forward_allocations.size());
  for (std::size_t index = 0U; index < forward_allocations.size(); ++index) {
    const ArenaAllocation& allocation = forward_allocations[index];
    if (allocation.buffer_ordinal() != index) {
      return lowering_invariant(
          "graph arena planner returned non-canonical allocation #b" +
          std::to_string(allocation.buffer_ordinal()) + " at index " +
          std::to_string(index));
    }
    placements.push_back(ArenaPlacement{
        allocation.buffer_ordinal(),
        allocation.offset_bytes(),
    });
  }

  auto reverse =
      GraphArenaPlacementVerifier::verify(source, placements, limits);
  if (!reverse.has_value()) {
    return lowering_invariant(
        "graph arena lowering failed reverse verification: " +
        std::string(error_code_name(reverse.error_if()->code)) + ": " +
        reverse.error_if()->message);
  }
  GraphArenaLoweringResult& verified = *reverse.value_if();

  if (verified.source_node_count() != source.nodes().size()) {
    return lowering_invariant(
        "graph arena forward and reverse source node counts disagree");
  }
  if (verified.execution_step_count() != execution_step_count) {
    return lowering_invariant(
        "graph arena forward and reverse execution step counts disagree");
  }
  if (verified.values_by_buffer_ordinal().size() != forward_values.size()) {
    return lowering_invariant(
        "graph arena forward and reverse value counts disagree");
  }
  if (verified.requests().size() != forward_requests.size()) {
    return lowering_invariant(
        "graph arena forward and reverse request counts disagree");
  }

  for (std::size_t index = 0U; index < forward_values.size(); ++index) {
    if (verified.values_by_buffer_ordinal()[index] != forward_values[index]) {
      return lowering_invariant(
          "graph arena forward and reverse value mappings disagree at #b" +
          std::to_string(index));
    }
    const auto expected_ordinal = static_cast<std::uint32_t>(index);
    if (verified.buffer_ordinal(forward_values[index]) != expected_ordinal) {
      return lowering_invariant(
          "graph arena reverse lookup disagrees at #b" +
          std::to_string(index));
    }
    if (verified.requests()[index] != forward_requests[index]) {
      return lowering_invariant(
          "graph arena forward and reverse requests disagree at #b" +
          std::to_string(index));
    }
  }

  const ArenaPlan& reverse_plan = verified.arena_plan();
  if (reverse_plan.limits() != limits) {
    return lowering_invariant(
        "graph arena reverse verifier changed the requested limits");
  }
  if (forward_plan.limits() != reverse_plan.limits()) {
    return lowering_invariant(
        "graph arena planned and reverse-verified limits disagree");
  }
  if (forward_plan.stats() != reverse_plan.stats()) {
    return lowering_invariant(
        "graph arena planned and reverse-verified statistics disagree");
  }
  if (forward_plan.allocations().size() !=
      reverse_plan.allocations().size()) {
    return lowering_invariant(
        "graph arena planned and reverse-verified allocation counts "
        "disagree");
  }
  for (std::size_t index = 0U; index < forward_allocations.size(); ++index) {
    if (!same_allocation(forward_allocations[index],
                         reverse_plan.allocations()[index])) {
      return lowering_invariant(
          "graph arena planned and reverse-verified allocations disagree at "
          "#b" +
          std::to_string(index));
    }
  }

  return reverse;
}

}  // namespace detail

Result<GraphArenaLoweringResult> GraphArenaLowering::run(
    const VerifiedGraph& source, const ArenaLimits limits) {
  const std::span<const Node> nodes = source.nodes();
  if (nodes.size() > static_cast<std::size_t>(
                         std::numeric_limits<std::uint32_t>::max())) {
    return lowering_invariant(
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
      return lowering_invariant(
          "graph arena source has a non-dense or foreign definition at index " +
          std::to_string(source_index));
    }
    if (forward_materializes_arena_buffer(node.operation())) {
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
  std::vector<ArenaBufferRequest> requests;
  values_by_buffer_ordinal.reserve(materialized_buffer_count);
  requests.reserve(materialized_buffer_count);

  std::uint64_t next_step = 0U;
  for (std::size_t source_index = 0U; source_index < nodes.size();
       ++source_index) {
    const Node& node = nodes[source_index];
    if (!forward_materializes_arena_buffer(node.operation())) {
      continue;
    }
    if (next_step >= static_cast<std::uint64_t>(
                         std::numeric_limits<std::uint32_t>::max())) {
      return lowering_invariant(
          "graph arena execution step count exceeds the uint32 domain");
    }
    if (values_by_buffer_ordinal.size() >=
        static_cast<std::size_t>(
            std::numeric_limits<std::uint32_t>::max())) {
      return lowering_invariant(
          "graph arena buffer count exceeds the uint32 domain");
    }

    const auto step = static_cast<std::uint32_t>(next_step);
    const auto consumer_boundary =
        static_cast<std::uint32_t>(next_step + 1U);
    for (const ValueId operand : node.inputs()) {
      const std::size_t operand_ordinal =
          static_cast<std::size_t>(operand.ordinal());
      if (operand_ordinal >= source_index || operand_ordinal >= nodes.size() ||
          source.type(operand) == nullptr ||
          nodes[operand_ordinal].output() != operand) {
        return lowering_invariant(
            "graph arena source has a foreign or forward operand %" +
            std::to_string(operand.ordinal()) + " at #n" +
            std::to_string(node.id().ordinal()));
      }
      const std::optional<std::uint32_t> operand_buffer =
          buffer_by_source_ordinal[operand_ordinal];
      if (operand_buffer.has_value()) {
        ArenaBufferRequest& request =
            requests[static_cast<std::size_t>(*operand_buffer)];
        request.live_end_step_exclusive =
            std::max(request.live_end_step_exclusive, consumer_boundary);
      } else if (forward_materializes_arena_buffer(
                     nodes[operand_ordinal].operation())) {
        return lowering_invariant(
            "graph arena operand mapping omitted a materialized value");
      }
    }

    const std::size_t node_payload_bytes = node.output_type().byte_count();
    if constexpr (sizeof(std::size_t) > sizeof(std::uint64_t)) {
      if (node_payload_bytes > static_cast<std::size_t>(
                                   std::numeric_limits<std::uint64_t>::max())) {
        return lowering_invariant(
            "graph arena tensor byte count exceeds the uint64 domain at #n" +
            std::to_string(node.id().ordinal()));
      }
    }
    const auto buffer_ordinal = static_cast<std::uint32_t>(
        values_by_buffer_ordinal.size());
    buffer_by_source_ordinal[source_index] = buffer_ordinal;
    values_by_buffer_ordinal.push_back(node.output());
    requests.push_back(ArenaBufferRequest{
        static_cast<std::uint64_t>(node_payload_bytes),
        step,
        consumer_boundary,
    });
    ++next_step;
  }

  const auto execution_step_count = static_cast<std::uint32_t>(next_step);
  for (std::size_t output_index = 0U;
       output_index < source.outputs().size(); ++output_index) {
    const GraphOutput& output = source.outputs()[output_index];
    const std::size_t source_ordinal =
        static_cast<std::size_t>(output.value().ordinal());
    if (output.id().ordinal != output_index ||
        source_ordinal >= nodes.size() ||
        source.type(output.value()) == nullptr ||
        nodes[source_ordinal].output() != output.value()) {
      return lowering_invariant(
          "graph arena source has an invalid output at index " +
          std::to_string(output_index));
    }
    const std::optional<std::uint32_t> buffer_ordinal =
        buffer_by_source_ordinal[source_ordinal];
    if (buffer_ordinal.has_value()) {
      ArenaBufferRequest& request =
          requests[static_cast<std::size_t>(*buffer_ordinal)];
      request.live_end_step_exclusive =
          std::max(request.live_end_step_exclusive, execution_step_count);
    } else if (forward_materializes_arena_buffer(
                   nodes[source_ordinal].operation())) {
      return lowering_invariant(
          "graph arena output mapping omitted a materialized value");
    }
  }

  auto planned = ArenaPlanner::run(requests, limits);
  if (!planned.has_value()) {
    return planner_failure(*planned.error_if());
  }
  const ArenaPlan& forward_plan = *planned.value_if();
  return detail::verify_graph_arena_lowering_agreement(
      source, values_by_buffer_ordinal, requests, execution_step_count,
      forward_plan, limits);
}

}  // namespace tensorkiln
