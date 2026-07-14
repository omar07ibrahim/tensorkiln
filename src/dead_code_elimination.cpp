#include "tensorkiln/dead_code_elimination.hpp"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace tensorkiln {
namespace {

template <class... Visitors>
struct Overloaded final : Visitors... {
  using Visitors::operator()...;
};

template <class... Visitors>
Overloaded(Visitors...) -> Overloaded<Visitors...>;

[[nodiscard]] Diagnostic invariant_error(std::string message) {
  return Diagnostic{
      ErrorCode::compiler_internal_invariant,
      std::move(message),
  };
}

[[nodiscard]] Diagnostic provenance_error(std::string message) {
  return Diagnostic{
      ErrorCode::provenance_domain_mismatch,
      std::move(message),
  };
}

[[nodiscard]] std::string node_label(const NodeId node) {
  return "#n" + std::to_string(node.ordinal());
}

[[nodiscard]] std::string value_label(const ValueId value) {
  return "%" + std::to_string(value.ordinal());
}

[[nodiscard]] Diagnostic replay_error(const Node& source,
                                      const Diagnostic& cause) {
  return invariant_error(
      "failed to replay " + node_label(source.id()) + ": " +
      std::string(error_code_name(cause.code)) + ": " + cause.message);
}

[[nodiscard]] Result<ValueId> arity_error(const Node& source,
                                          const std::size_t expected) {
  return Result<ValueId>::failure(invariant_error(
      node_label(source.id()) + " has " +
      std::to_string(source.inputs().size()) + " operands; expected " +
      std::to_string(expected)));
}

[[nodiscard]] bool same_float_bits(const std::span<const float> left,
                                   const std::span<const float> right) {
  if (left.size() != right.size()) {
    return false;
  }
  for (std::size_t index = 0U; index < left.size(); ++index) {
    if (std::bit_cast<std::uint32_t>(left[index]) !=
        std::bit_cast<std::uint32_t>(right[index])) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] Result<ValueId> replay_operation(
    GraphBuilder& builder, const Node& source,
    const std::span<const ValueId> operands) {
  auto result = std::visit(
      Overloaded{
          [&builder, &source, operands](const InputOp& operation) {
            if (!operands.empty()) {
              return arity_error(source, 0U);
            }
            return builder.input(operation.name, source.output_type());
          },
          [&builder, &source, operands](const ConstantOp& operation) {
            if (!operands.empty()) {
              return arity_error(source, 0U);
            }
            return builder.constant(
                operation.name, source.output_type(),
                std::span<const float>{operation.data});
          },
          [&builder, &source, operands](const AddOp&) {
            if (operands.size() != 2U) {
              return arity_error(source, 2U);
            }
            return builder.add(operands[0], operands[1]);
          },
          [&builder, &source, operands](const MatMulOp&) {
            if (operands.size() != 2U) {
              return arity_error(source, 2U);
            }
            return builder.matmul(operands[0], operands[1]);
          },
          [&builder, &source, operands](const ReluOp&) {
            if (operands.size() != 1U) {
              return arity_error(source, 1U);
            }
            return builder.relu(operands[0]);
          },
      },
      source.operation());

  if (!result.has_value()) {
    return Result<ValueId>::failure(replay_error(source, *result.error_if()));
  }
  return result;
}

[[nodiscard]] std::optional<Diagnostic> validate_replayed_node(
    const Node& source, const Node& result,
    const std::span<const std::optional<ValueId>> value_map) {
  if (source.output_type() != result.output_type()) {
    return invariant_error(
        "replayed " + node_label(source.id()) + " changed output type from " +
        source.output_type().to_string() + " to " +
        result.output_type().to_string());
  }
  if (source.inputs().size() != result.inputs().size()) {
    return invariant_error("replayed " + node_label(source.id()) +
                           " changed operand count");
  }
  for (std::size_t index = 0U; index < source.inputs().size(); ++index) {
    const std::size_t source_ordinal =
        static_cast<std::size_t>(source.inputs()[index].ordinal());
    if (source_ordinal >= value_map.size() ||
        !value_map[source_ordinal].has_value() ||
        *value_map[source_ordinal] != result.inputs()[index]) {
      return invariant_error("replayed " + node_label(source.id()) +
                             " changed operand topology");
    }
  }
  if (source.operation().index() != result.operation().index()) {
    return invariant_error("replayed " + node_label(source.id()) +
                           " changed operation kind");
  }

  const bool operation_matches = std::visit(
      Overloaded{
          [&result](const InputOp& source_operation) {
            return source_operation.name ==
                   std::get<InputOp>(result.operation()).name;
          },
          [&result](const ConstantOp& source_operation) {
            const ConstantOp& result_operation =
                std::get<ConstantOp>(result.operation());
            return source_operation.name == result_operation.name &&
                   source_operation.fingerprint ==
                       result_operation.fingerprint &&
                   same_float_bits(source_operation.data,
                                   result_operation.data);
          },
          [](const AddOp&) { return true; },
          [](const MatMulOp&) { return true; },
          [](const ReluOp&) { return true; },
      },
      source.operation());
  if (!operation_matches) {
    return invariant_error("replayed " + node_label(source.id()) +
                           " changed operation payload");
  }
  return std::nullopt;
}

void append_source_set(std::string& destination,
                       const std::span<const SourceDefinition> sources) {
  destination += "{";
  for (std::size_t index = 0U; index < sources.size(); ++index) {
    if (index != 0U) {
      destination += ", ";
    }
    destination += "source " + node_label(sources[index].node()) + " " +
                   value_label(sources[index].value());
  }
  destination += "}";
}

void append_provenance_entry(std::string& destination,
                             const NodeProvenance& entry) {
  destination += "  " + node_label(entry.result_node()) + " " +
                 value_label(entry.result_value()) + " <- ";
  append_source_set(destination, entry.sources());
  destination += "\n";
}

[[nodiscard]] std::optional<Diagnostic> validate_source_provenance_domain(
    const VerifiedGraph& source, const GraphProvenance& provenance) {
  if (provenance.entries().size() != source.nodes().size()) {
    return provenance_error(
        "source provenance has " +
        std::to_string(provenance.entries().size()) +
        " entries for a graph with " + std::to_string(source.nodes().size()) +
        " nodes");
  }
  for (const Node& definition : source.nodes()) {
    const NodeProvenance* by_node = provenance.for_result(definition.id());
    const NodeProvenance* by_value =
        provenance.for_result(definition.output());
    if (by_node == nullptr || by_value == nullptr || by_node != by_value) {
      return provenance_error(
          "source provenance does not describe " +
          node_label(definition.id()) + " " +
          value_label(definition.output()));
    }
  }
  return std::nullopt;
}

}  // namespace

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
    append_provenance_entry(result, entry);
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
      return Result<DeadCodeEliminationResult>::failure(invariant_error(
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
      return Result<DeadCodeEliminationResult>::failure(invariant_error(
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
        return Result<DeadCodeEliminationResult>::failure(invariant_error(
            node_label(source_nodes[index].id()) +
            " has a foreign or non-topological operand " +
            value_label(operand)));
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
        return Result<DeadCodeEliminationResult>::failure(invariant_error(
            "no replayed definition exists for " + value_label(operand)));
      }
      operands.push_back(*value_map[operand_index]);
    }

    auto replayed = replay_operation(builder, definition, operands);
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
      return Result<DeadCodeEliminationResult>::failure(invariant_error(
          "no replayed definition exists for output @" + output.name()));
    }
    auto replayed_output = builder.output(output.name(), *value_map[source_ordinal]);
    if (!replayed_output.has_value()) {
      return Result<DeadCodeEliminationResult>::failure(invariant_error(
          "failed to replay output @" + output.name() + ": " +
          std::string(error_code_name(replayed_output.error_if()->code)) +
          ": " + replayed_output.error_if()->message));
    }
  }

  auto finished = std::move(builder).finish();
  if (!finished.has_value()) {
    return Result<DeadCodeEliminationResult>::failure(invariant_error(
        "failed to finish replayed graph: " +
        std::string(error_code_name(finished.error_if()->code)) + ": " +
        finished.error_if()->message));
  }
  VerifiedGraph graph = std::move(*finished.value_if());

  if (graph.limits() != source.limits() ||
      graph.nodes().size() != retained_source_ordinals.size() ||
      graph.outputs().size() != source.outputs().size()) {
    return Result<DeadCodeEliminationResult>::failure(invariant_error(
        "replayed graph changed construction limits or declaration counts"));
  }

  std::vector<NodeProvenance> provenance_entries;
  provenance_entries.reserve(graph.nodes().size());
  for (std::size_t result_index = 0U; result_index < graph.nodes().size();
       ++result_index) {
    const Node& result_definition = graph.nodes()[result_index];
    const Node& source_definition =
        source_nodes[retained_source_ordinals[result_index]];
    if (const auto error =
            validate_replayed_node(source_definition, result_definition,
                                   value_map)) {
      return Result<DeadCodeEliminationResult>::failure(*error);
    }
    std::vector<SourceDefinition> sources;
    sources.push_back(
        SourceDefinition(source_definition.id(), source_definition.output()));
    provenance_entries.push_back(NodeProvenance(
        result_definition.id(), result_definition.output(),
        std::move(sources)));
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
      return Result<DeadCodeEliminationResult>::failure(invariant_error(
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
  return Result<DeadCodeEliminationResult>::success(
      DeadCodeEliminationResult(
          std::move(graph), GraphProvenance(std::move(provenance_entries)),
          stats));
}

Result<DeadCodeEliminationResult> DeadCodeElimination::run(
    const VerifiedGraph& source,
    const GraphProvenance& source_provenance) {
  if (const auto error =
          validate_source_provenance_domain(source, source_provenance)) {
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
