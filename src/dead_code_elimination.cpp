#include "tensorkiln/dead_code_elimination.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "compiler_support.hpp"

namespace tensorkiln {

DeadCodeEliminationResult::DeadCodeEliminationResult(
    VerifiedGraph graph, GraphProvenance provenance,
    const DeadCodeEliminationStats stats)
    : graph_(std::move(graph)),
      provenance_(std::move(provenance)),
      stats_(stats) {}

std::string DeadCodeEliminationResult::dump() const {
  std::string result{"tensorkiln.dce v0 {\n"};
  result += "  nodes {source=" + std::to_string(stats_.source_nodes) +
            ", retained=" + std::to_string(stats_.retained_nodes) +
            ", removed=" + std::to_string(stats_.removed_nodes) + "}\n";
  result += "  constant_elements {source=" +
            std::to_string(stats_.source_constant_elements) +
            ", retained=" +
            std::to_string(stats_.retained_constant_elements) +
            ", removed=" +
            std::to_string(stats_.removed_constant_elements) + "}\n";
  for (const NodeProvenance& entry : provenance_.entries()) {
    detail::append_provenance_entry(result, entry);
  }
  result += "}\n";
  return result;
}

Result<DeadCodeEliminationResult> DeadCodeElimination::run(
    const VerifiedGraph& source) {
  const std::span<const Node> source_nodes = source.nodes();
  std::vector<std::uint8_t> live(source_nodes.size(), UINT8_C(0));

  std::uint64_t source_constant_elements = 0U;
  for (std::size_t index = 0U; index < source_nodes.size(); ++index) {
    const Node& definition = source_nodes[index];
    if (definition.id().ordinal() != index ||
        definition.output().ordinal() != index) {
      return Result<DeadCodeEliminationResult>::failure(
          detail::invariant_error(
              "verified graph definitions are not dense at ordinal " +
              std::to_string(index)));
    }
    if (std::holds_alternative<InputOp>(definition.operation())) {
      live[index] = UINT8_C(1);
    }
    if (const auto* constant =
            std::get_if<ConstantOp>(&definition.operation())) {
      source_constant_elements +=
          static_cast<std::uint64_t>(constant->data.size());
    }
  }

  for (const GraphOutput& output : source.outputs()) {
    if (source.type(output.value()) == nullptr) {
      return Result<DeadCodeEliminationResult>::failure(
          detail::invariant_error(
              "graph output " + std::to_string(output.id().ordinal) +
              " references a foreign value"));
    }
    live[static_cast<std::size_t>(output.value().ordinal())] = UINT8_C(1);
  }

  for (std::size_t reverse = source_nodes.size(); reverse > 0U; --reverse) {
    const std::size_t index = reverse - 1U;
    if (live[index] == UINT8_C(0)) {
      continue;
    }
    for (const ValueId operand : source_nodes[index].inputs()) {
      if (source.type(operand) == nullptr ||
          static_cast<std::size_t>(operand.ordinal()) >= index) {
        return Result<DeadCodeEliminationResult>::failure(
            detail::invariant_error(
                detail::node_label(source_nodes[index].id()) +
                " has a foreign or non-topological operand " +
                detail::value_label(operand)));
      }
      live[static_cast<std::size_t>(operand.ordinal())] = UINT8_C(1);
    }
  }

  GraphBuilder builder(source.limits());
  std::vector<std::optional<ValueId>> value_map(source_nodes.size());
  std::vector<std::size_t> retained_source_ordinals;
  retained_source_ordinals.reserve(source_nodes.size());
  std::uint64_t retained_constant_elements = 0U;

  for (std::size_t index = 0U; index < source_nodes.size(); ++index) {
    if (live[index] == UINT8_C(0)) {
      continue;
    }
    const Node& definition = source_nodes[index];
    std::vector<ValueId> operands;
    operands.reserve(definition.inputs().size());
    for (const ValueId operand : definition.inputs()) {
      const std::size_t operand_index =
          static_cast<std::size_t>(operand.ordinal());
      if (operand_index >= value_map.size() ||
          !value_map[operand_index].has_value()) {
        return Result<DeadCodeEliminationResult>::failure(
            detail::invariant_error(
                "no replayed definition exists for " +
                detail::value_label(operand)));
      }
      operands.push_back(*value_map[operand_index]);
    }

    auto replayed = detail::replay_operation(builder, definition, operands);
    if (!replayed.has_value()) {
      return Result<DeadCodeEliminationResult>::failure(*replayed.error_if());
    }
    value_map[index] = *replayed.value_if();
    retained_source_ordinals.push_back(index);
    if (const auto* constant =
            std::get_if<ConstantOp>(&definition.operation())) {
      retained_constant_elements +=
          static_cast<std::uint64_t>(constant->data.size());
    }
  }

  for (const GraphOutput& output : source.outputs()) {
    const std::size_t source_ordinal =
        static_cast<std::size_t>(output.value().ordinal());
    if (source_ordinal >= value_map.size() ||
        !value_map[source_ordinal].has_value()) {
      return Result<DeadCodeEliminationResult>::failure(
          detail::invariant_error(
              "no replayed definition exists for output @" + output.name()));
    }
    auto replayed_output =
        builder.output(output.name(), *value_map[source_ordinal]);
    if (!replayed_output.has_value()) {
      return Result<DeadCodeEliminationResult>::failure(
          detail::invariant_error(
              "failed to replay output @" + output.name() + ": " +
              std::string(error_code_name(replayed_output.error_if()->code)) +
              ": " + replayed_output.error_if()->message));
    }
  }

  auto finished = std::move(builder).finish();
  if (!finished.has_value()) {
    return Result<DeadCodeEliminationResult>::failure(
        detail::invariant_error(
            "failed to finish replayed graph: " +
            std::string(error_code_name(finished.error_if()->code)) + ": " +
            finished.error_if()->message));
  }
  VerifiedGraph graph = std::move(*finished.value_if());

  if (graph.limits() != source.limits() ||
      graph.nodes().size() != retained_source_ordinals.size() ||
      graph.outputs().size() != source.outputs().size()) {
    return Result<DeadCodeEliminationResult>::failure(
        detail::invariant_error(
            "replayed graph changed construction limits or declaration "
            "counts"));
  }

  std::vector<std::vector<NodeId>> source_nodes_by_result;
  source_nodes_by_result.reserve(graph.nodes().size());
  for (std::size_t result_index = 0U; result_index < graph.nodes().size();
       ++result_index) {
    const Node& result_definition = graph.nodes()[result_index];
    const Node& source_definition =
        source_nodes[retained_source_ordinals[result_index]];
    if (const auto error =
            detail::validate_replayed_node(source_definition,
                                           result_definition, value_map)) {
      return Result<DeadCodeEliminationResult>::failure(*error);
    }
    source_nodes_by_result.push_back(
        std::vector<NodeId>{source_definition.id()});
  }

  for (std::size_t index = 0U; index < graph.outputs().size(); ++index) {
    const GraphOutput& source_output = source.outputs()[index];
    const GraphOutput& result_output = graph.outputs()[index];
    const std::size_t source_ordinal =
        static_cast<std::size_t>(source_output.value().ordinal());
    if (source_output.id() != result_output.id() ||
        source_output.name() != result_output.name() ||
        !value_map[source_ordinal].has_value() ||
        *value_map[source_ordinal] != result_output.value()) {
      return Result<DeadCodeEliminationResult>::failure(
          detail::invariant_error(
              "replayed graph changed output declaration " +
              std::to_string(index)));
    }
  }

  const auto source_node_count =
      static_cast<std::uint32_t>(source_nodes.size());
  const auto retained_node_count =
      static_cast<std::uint32_t>(graph.nodes().size());
  const DeadCodeEliminationStats stats{
      source_node_count,
      retained_node_count,
      source_node_count - retained_node_count,
      source_constant_elements,
      retained_constant_elements,
      source_constant_elements - retained_constant_elements,
  };
  auto provenance = GraphProvenance::create(
      graph, source, std::move(source_nodes_by_result));
  if (!provenance.has_value()) {
    return Result<DeadCodeEliminationResult>::failure(
        detail::invariant_error(
            "failed to construct DCE provenance: " +
            std::string(error_code_name(provenance.error_if()->code)) + ": " +
            provenance.error_if()->message));
  }
  return Result<DeadCodeEliminationResult>::success(
      DeadCodeEliminationResult(
          std::move(graph), std::move(*provenance.value_if()), stats));
}

Result<DeadCodeEliminationResult> DeadCodeElimination::run(
    const VerifiedGraph& source,
    const GraphProvenance& source_provenance) {
  if (const auto error =
          detail::validate_source_provenance_domain(source,
                                                    source_provenance)) {
    return Result<DeadCodeEliminationResult>::failure(*error);
  }

  auto immediate = run(source);
  if (!immediate.has_value()) {
    return Result<DeadCodeEliminationResult>::failure(*immediate.error_if());
  }
  DeadCodeEliminationResult& immediate_result = *immediate.value_if();
  auto composed = immediate_result.provenance_.compose(source_provenance);
  if (!composed.has_value()) {
    return Result<DeadCodeEliminationResult>::failure(*composed.error_if());
  }

  return Result<DeadCodeEliminationResult>::success(
      DeadCodeEliminationResult(
          std::move(immediate_result.graph_),
          std::move(*composed.value_if()), immediate_result.stats_));
}

}  // namespace tensorkiln
