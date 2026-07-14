#include "compiler_support.hpp"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <variant>

namespace tensorkiln::detail {
namespace {

template <class... Visitors>
struct Overloaded final : Visitors... {
  using Visitors::operator()...;
};

template <class... Visitors>
Overloaded(Visitors...) -> Overloaded<Visitors...>;

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

}  // namespace

Diagnostic invariant_error(std::string message) {
  return Diagnostic{
      ErrorCode::compiler_internal_invariant,
      std::move(message),
  };
}

Diagnostic provenance_error(std::string message) {
  return Diagnostic{
      ErrorCode::provenance_domain_mismatch,
      std::move(message),
  };
}

std::string node_label(const NodeId node) {
  return "#n" + std::to_string(node.ordinal());
}

std::string value_label(const ValueId value) {
  return "%" + std::to_string(value.ordinal());
}

Result<ValueId> replay_operation(
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

std::optional<Diagnostic> validate_replayed_node(
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

void append_provenance_entry(std::string& destination,
                             const NodeProvenance& entry) {
  destination += "  " + node_label(entry.result_node()) + " " +
                 value_label(entry.result_value()) + " <- ";
  append_source_set(destination, entry.sources());
  destination += "\n";
}

std::optional<Diagnostic> validate_source_provenance_domain(
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

}  // namespace tensorkiln::detail
