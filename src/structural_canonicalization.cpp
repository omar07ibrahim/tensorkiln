#include "tensorkiln/structural_canonicalization.hpp"

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "compiler_support.hpp"

namespace tensorkiln {
namespace {

enum class CanonicalOperation : std::uint8_t {
  add,
  matmul,
  relu,
};

enum class SourceDisposition : std::uint8_t {
  emitted,
  common_subexpression,
  redundant_relu,
};

struct CanonicalKey final {
  CanonicalOperation operation;
  std::vector<std::uint32_t> operands;
  std::uint8_t element_type;
  std::vector<std::int64_t> extents;
  std::uint64_t element_count;
  std::size_t byte_count;

  friend bool operator==(const CanonicalKey&,
                         const CanonicalKey&) noexcept = default;

  friend bool operator<(const CanonicalKey& left,
                        const CanonicalKey& right) noexcept {
    return std::tie(left.operation, left.operands, left.element_type,
                    left.extents, left.element_count, left.byte_count) <
           std::tie(right.operation, right.operands, right.element_type,
                    right.extents, right.element_count, right.byte_count);
  }
};

[[nodiscard]] std::optional<CanonicalOperation> canonical_operation(
    const Operation& operation) noexcept {
  if (std::holds_alternative<AddOp>(operation)) {
    return CanonicalOperation::add;
  }
  if (std::holds_alternative<MatMulOp>(operation)) {
    return CanonicalOperation::matmul;
  }
  if (std::holds_alternative<ReluOp>(operation)) {
    return CanonicalOperation::relu;
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<CanonicalKey> make_key(
    const Node& definition, const std::span<const ValueId> operands) {
  const std::optional<CanonicalOperation> operation =
      canonical_operation(definition.operation());
  if (!operation.has_value()) {
    return std::nullopt;
  }

  std::vector<std::uint32_t> operand_ordinals;
  operand_ordinals.reserve(operands.size());
  for (const ValueId operand : operands) {
    operand_ordinals.push_back(operand.ordinal());
  }

  const TensorType& type = definition.output_type();
  const std::span<const std::int64_t> type_extents = type.shape().extents();
  return CanonicalKey{
      *operation,
      std::move(operand_ordinals),
      static_cast<std::uint8_t>(type.element_type()),
      std::vector<std::int64_t>(type_extents.begin(), type_extents.end()),
      type.numel(),
      type.byte_count(),
  };
}

[[nodiscard]] std::optional<Diagnostic> validate_output_contract(
    const VerifiedGraph& source, const VerifiedGraph& result,
    const std::span<const std::optional<ValueId>> value_map) {
  if (result.outputs().size() != source.outputs().size()) {
    return detail::invariant_error(
        "canonicalization changed the output declaration count");
  }

  std::vector<std::optional<std::uint32_t>> result_for_source(
      source.nodes().size());
  std::vector<std::optional<std::uint32_t>> source_for_result(
      result.nodes().size());
  for (std::size_t index = 0U; index < source.outputs().size(); ++index) {
    const GraphOutput& source_output = source.outputs()[index];
    const GraphOutput& result_output = result.outputs()[index];
    const std::size_t source_ordinal =
        static_cast<std::size_t>(source_output.value().ordinal());
    const std::size_t result_ordinal =
        static_cast<std::size_t>(result_output.value().ordinal());
    if (source_ordinal >= value_map.size() ||
        result_ordinal >= result.nodes().size() ||
        !value_map[source_ordinal].has_value() ||
        source_output.id() != result_output.id() ||
        source_output.name() != result_output.name() ||
        *value_map[source_ordinal] != result_output.value()) {
      return detail::invariant_error(
          "canonicalization changed output declaration " +
          std::to_string(index));
    }

    const std::uint32_t source_value = source_output.value().ordinal();
    const std::uint32_t result_value = result_output.value().ordinal();
    if ((result_for_source[source_ordinal].has_value() &&
         *result_for_source[source_ordinal] != result_value) ||
        (source_for_result[result_ordinal].has_value() &&
         *source_for_result[result_ordinal] != source_value)) {
      return detail::invariant_error(
          "canonicalization changed output alias classes");
    }
    result_for_source[source_ordinal] = result_value;
    source_for_result[result_ordinal] = source_value;
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<Diagnostic> validate_source_mapping(
    const VerifiedGraph& source, const VerifiedGraph& result,
    const std::span<const std::optional<ValueId>> value_map,
    const std::span<const SourceDisposition> dispositions,
    const std::span<const std::size_t> representatives) {
  for (std::size_t result_index = 0U;
       result_index < result.nodes().size(); ++result_index) {
    const Node& source_definition =
        source.nodes()[representatives[result_index]];
    if (const auto error = detail::validate_replayed_node(
            source_definition, result.nodes()[result_index], value_map)) {
      return error;
    }
  }

  for (std::size_t source_index = 0U;
       source_index < source.nodes().size(); ++source_index) {
    const Node& source_definition = source.nodes()[source_index];
    if (!value_map[source_index].has_value()) {
      return detail::invariant_error(
          "canonicalization did not map " +
          detail::node_label(source_definition.id()));
    }
    const ValueId result_value = *value_map[source_index];
    const std::size_t result_index =
        static_cast<std::size_t>(result_value.ordinal());
    if (result_index >= result.nodes().size() ||
        result.nodes()[result_index].output() != result_value ||
        result.nodes()[result_index].output_type() !=
            source_definition.output_type()) {
      return detail::invariant_error(
          "canonicalization changed the type or owner of " +
          detail::node_label(source_definition.id()));
    }

    if (dispositions[source_index] == SourceDisposition::emitted) {
      continue;
    }

    std::vector<ValueId> operands;
    operands.reserve(source_definition.inputs().size());
    for (const ValueId operand : source_definition.inputs()) {
      const std::size_t operand_index =
          static_cast<std::size_t>(operand.ordinal());
      if (operand_index >= value_map.size() ||
          !value_map[operand_index].has_value()) {
        return detail::invariant_error(
            "canonicalization lost a merged operand");
      }
      operands.push_back(*value_map[operand_index]);
    }

    if (dispositions[source_index] ==
        SourceDisposition::common_subexpression) {
      const std::optional<CanonicalKey> source_key =
          make_key(source_definition, operands);
      const std::optional<CanonicalKey> result_key = make_key(
          result.nodes()[result_index],
          result.nodes()[result_index].inputs());
      if (!source_key.has_value() || !result_key.has_value() ||
          *source_key != *result_key) {
        return detail::invariant_error(
            "canonicalization merged non-equivalent definitions");
      }
      continue;
    }

    if (!std::holds_alternative<ReluOp>(source_definition.operation()) ||
        operands.size() != 1U || operands[0] != result_value ||
        !std::holds_alternative<ReluOp>(
            result.nodes()[result_index].operation())) {
      return detail::invariant_error(
          "canonicalization applied an invalid redundant ReLU rewrite");
    }
  }
  return std::nullopt;
}

}  // namespace

StructuralCanonicalizationResult::StructuralCanonicalizationResult(
    VerifiedGraph graph, GraphProvenance provenance,
    const StructuralCanonicalizationStats stats)
    : graph_(std::move(graph)),
      provenance_(std::move(provenance)),
      stats_(stats) {}

std::string StructuralCanonicalizationResult::dump() const {
  std::string result{"tensorkiln.structural_canonicalization v0 {\n"};
  result += "  nodes {source=" + std::to_string(stats_.source_nodes) +
            ", result=" + std::to_string(stats_.result_nodes) +
            ", merged=" + std::to_string(stats_.merged_nodes) + "}\n";
  result += "  merges {common_subexpressions=" +
            std::to_string(stats_.common_subexpressions) +
            ", redundant_relus=" +
            std::to_string(stats_.redundant_relus) + "}\n";
  result += "  guards {preserved_output_distinctions=" +
            std::to_string(stats_.preserved_output_distinctions) + "}\n";
  for (const NodeProvenance& entry : provenance_.entries()) {
    detail::append_provenance_entry(result, entry);
  }
  result += "}\n";
  return result;
}

Result<StructuralCanonicalizationResult> StructuralCanonicalization::run(
    const VerifiedGraph& source) {
  const std::span<const Node> source_nodes = source.nodes();
  std::vector<std::uint8_t> source_is_output(source_nodes.size(), UINT8_C(0));

  for (std::size_t index = 0U; index < source_nodes.size(); ++index) {
    const Node& definition = source_nodes[index];
    if (definition.id().ordinal() != index ||
        definition.output().ordinal() != index ||
        source.type(definition.output()) == nullptr) {
      return Result<StructuralCanonicalizationResult>::failure(
          detail::invariant_error(
              "verified graph definitions are not dense at ordinal " +
              std::to_string(index)));
    }
  }
  for (const GraphOutput& output : source.outputs()) {
    const std::size_t source_index =
        static_cast<std::size_t>(output.value().ordinal());
    if (source_index >= source_nodes.size() ||
        source.type(output.value()) == nullptr) {
      return Result<StructuralCanonicalizationResult>::failure(
          detail::invariant_error(
              "graph output " + std::to_string(output.id().ordinal) +
              " references a foreign value"));
    }
    source_is_output[source_index] = UINT8_C(1);
  }

  GraphBuilder builder(source.limits());
  std::vector<std::optional<ValueId>> value_map(source_nodes.size());
  std::vector<SourceDisposition> dispositions(
      source_nodes.size(), SourceDisposition::emitted);
  std::vector<std::vector<NodeId>> source_nodes_by_result;
  source_nodes_by_result.reserve(source_nodes.size());
  std::vector<std::size_t> representatives;
  representatives.reserve(source_nodes.size());
  std::vector<std::uint8_t> result_is_relu;
  result_is_relu.reserve(source_nodes.size());
  std::vector<std::uint8_t> result_has_output_source;
  result_has_output_source.reserve(source_nodes.size());
  std::vector<TensorType> result_types;
  result_types.reserve(source_nodes.size());
  std::map<CanonicalKey, ValueId> candidates;

  std::uint32_t common_subexpressions = 0U;
  std::uint32_t redundant_relus = 0U;
  std::uint32_t preserved_output_distinctions = 0U;

  for (std::size_t index = 0U; index < source_nodes.size(); ++index) {
    const Node& definition = source_nodes[index];
    std::vector<ValueId> operands;
    operands.reserve(definition.inputs().size());
    for (const ValueId operand : definition.inputs()) {
      const std::size_t operand_index =
          static_cast<std::size_t>(operand.ordinal());
      if (source.type(operand) == nullptr || operand_index >= index ||
          operand_index >= value_map.size() ||
          !value_map[operand_index].has_value()) {
        return Result<StructuralCanonicalizationResult>::failure(
            detail::invariant_error(
                detail::node_label(definition.id()) +
                " has a foreign, non-topological, or unmapped operand " +
                detail::value_label(operand)));
      }
      operands.push_back(*value_map[operand_index]);
    }

    const bool is_output = source_is_output[index] != UINT8_C(0);
    bool equivalent_candidate_seen = false;

    if (std::holds_alternative<ReluOp>(definition.operation()) &&
        operands.size() == 1U) {
      const ValueId operand = operands[0];
      const std::size_t operand_index =
          static_cast<std::size_t>(operand.ordinal());
      if (operand_index >= result_is_relu.size()) {
        return Result<StructuralCanonicalizationResult>::failure(
            detail::invariant_error(
                "canonical ReLU operand does not belong to the result "
                "graph"));
      }
      if (result_is_relu[operand_index] != UINT8_C(0)) {
        equivalent_candidate_seen = true;
        if (!is_output ||
            result_has_output_source[operand_index] == UINT8_C(0)) {
          value_map[index] = operand;
          dispositions[index] = SourceDisposition::redundant_relu;
          source_nodes_by_result[operand_index].push_back(definition.id());
          if (is_output) {
            result_has_output_source[operand_index] = UINT8_C(1);
          }
          ++redundant_relus;
          continue;
        }
      }
    }

    std::optional<CanonicalKey> key = make_key(definition, operands);
    if (key.has_value()) {
      const auto candidate_group = candidates.find(*key);
      if (candidate_group != candidates.end()) {
        equivalent_candidate_seen = true;
        const ValueId candidate = candidate_group->second;
        const std::size_t candidate_index =
            static_cast<std::size_t>(candidate.ordinal());
        if (candidate_index >= result_types.size()) {
          return Result<StructuralCanonicalizationResult>::failure(
              detail::invariant_error(
                  "canonical candidate does not belong to the result "
                  "graph"));
        }
        if (result_types[candidate_index] != definition.output_type()) {
          return Result<StructuralCanonicalizationResult>::failure(
              detail::invariant_error(
                  "canonical key admitted incompatible tensor types"));
        }
        if (!is_output ||
            result_has_output_source[candidate_index] == UINT8_C(0)) {
          value_map[index] = candidate;
          dispositions[index] = SourceDisposition::common_subexpression;
          source_nodes_by_result[candidate_index].push_back(definition.id());
          if (is_output) {
            result_has_output_source[candidate_index] = UINT8_C(1);
          }
          ++common_subexpressions;
          continue;
        }
      }
    }

    if (equivalent_candidate_seen) {
      ++preserved_output_distinctions;
    }

    auto replayed = detail::replay_operation(builder, definition, operands);
    if (!replayed.has_value()) {
      return Result<StructuralCanonicalizationResult>::failure(
          *replayed.error_if());
    }
    const ValueId replayed_value = *replayed.value_if();
    const std::size_t result_index =
        static_cast<std::size_t>(replayed_value.ordinal());
    if (result_index != source_nodes_by_result.size()) {
      return Result<StructuralCanonicalizationResult>::failure(
          detail::invariant_error(
              "canonical graph definitions are not dense"));
    }

    value_map[index] = replayed_value;
    source_nodes_by_result.push_back(
        std::vector<NodeId>{definition.id()});
    representatives.push_back(index);
    result_is_relu.push_back(
        std::holds_alternative<ReluOp>(definition.operation()) ? UINT8_C(1)
                                                              : UINT8_C(0));
    result_has_output_source.push_back(is_output ? UINT8_C(1) : UINT8_C(0));
    result_types.push_back(definition.output_type());
    if (key.has_value()) {
      // A blocked duplicate is emitted only because it is an output itself,
      // so it cannot be a compatible candidate for any later output.
      candidates.emplace(*key, replayed_value);
    }
  }

  for (const GraphOutput& output : source.outputs()) {
    const std::size_t source_index =
        static_cast<std::size_t>(output.value().ordinal());
    if (source_index >= value_map.size() ||
        !value_map[source_index].has_value()) {
      return Result<StructuralCanonicalizationResult>::failure(
          detail::invariant_error(
              "no canonical definition exists for output @" +
              output.name()));
    }
    auto replayed_output =
        builder.output(output.name(), *value_map[source_index]);
    if (!replayed_output.has_value()) {
      return Result<StructuralCanonicalizationResult>::failure(
          detail::invariant_error(
              "failed to replay output @" + output.name() + ": " +
              std::string(error_code_name(replayed_output.error_if()->code)) +
              ": " + replayed_output.error_if()->message));
    }
  }

  auto finished = std::move(builder).finish();
  if (!finished.has_value()) {
    return Result<StructuralCanonicalizationResult>::failure(
        detail::invariant_error(
            "failed to finish canonical graph: " +
            std::string(error_code_name(finished.error_if()->code)) + ": " +
            finished.error_if()->message));
  }
  VerifiedGraph graph = std::move(*finished.value_if());

  const auto source_node_count =
      static_cast<std::uint32_t>(source_nodes.size());
  const auto result_node_count =
      static_cast<std::uint32_t>(graph.nodes().size());
  const std::uint32_t merged_node_count =
      source_node_count - result_node_count;
  if (graph.limits() != source.limits() ||
      graph.nodes().size() != source_nodes_by_result.size() ||
      graph.nodes().size() != representatives.size() ||
      merged_node_count != common_subexpressions + redundant_relus) {
    return Result<StructuralCanonicalizationResult>::failure(
        detail::invariant_error(
            "canonical graph changed limits or violated merge accounting"));
  }

  if (const auto error = validate_source_mapping(
          source, graph, value_map, dispositions, representatives)) {
    return Result<StructuralCanonicalizationResult>::failure(*error);
  }
  if (const auto error = validate_output_contract(source, graph, value_map)) {
    return Result<StructuralCanonicalizationResult>::failure(*error);
  }

  std::size_t mapped_source_count = 0U;
  for (const std::vector<NodeId>& sources : source_nodes_by_result) {
    mapped_source_count += sources.size();
  }
  if (mapped_source_count != source_nodes.size()) {
    return Result<StructuralCanonicalizationResult>::failure(
        detail::invariant_error(
            "canonical provenance does not cover every source definition"));
  }

  auto provenance = GraphProvenance::create(
      graph, source, std::move(source_nodes_by_result));
  if (!provenance.has_value()) {
    return Result<StructuralCanonicalizationResult>::failure(
        detail::invariant_error(
            "failed to construct canonicalization provenance: " +
            std::string(error_code_name(provenance.error_if()->code)) + ": " +
            provenance.error_if()->message));
  }

  const StructuralCanonicalizationStats stats{
      source_node_count,
      result_node_count,
      merged_node_count,
      common_subexpressions,
      redundant_relus,
      preserved_output_distinctions,
  };
  return Result<StructuralCanonicalizationResult>::success(
      StructuralCanonicalizationResult(
          std::move(graph), std::move(*provenance.value_if()), stats));
}

Result<StructuralCanonicalizationResult> StructuralCanonicalization::run(
    const VerifiedGraph& source,
    const GraphProvenance& source_provenance) {
  if (const auto error = detail::validate_source_provenance_domain(
          source, source_provenance)) {
    return Result<StructuralCanonicalizationResult>::failure(*error);
  }

  auto immediate = run(source);
  if (!immediate.has_value()) {
    return Result<StructuralCanonicalizationResult>::failure(
        *immediate.error_if());
  }
  StructuralCanonicalizationResult& immediate_result =
      *immediate.value_if();
  auto composed =
      immediate_result.provenance_.compose(source_provenance);
  if (!composed.has_value()) {
    return Result<StructuralCanonicalizationResult>::failure(
        *composed.error_if());
  }

  return Result<StructuralCanonicalizationResult>::success(
      StructuralCanonicalizationResult(
          std::move(immediate_result.graph_),
          std::move(*composed.value_if()), immediate_result.stats_));
}

}  // namespace tensorkiln
