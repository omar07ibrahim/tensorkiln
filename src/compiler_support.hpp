#pragma once

#include <optional>
#include <span>
#include <string>

#include "tensorkiln/provenance.hpp"

namespace tensorkiln::detail {

[[nodiscard]] Diagnostic invariant_error(std::string message);
[[nodiscard]] Diagnostic provenance_error(std::string message);

[[nodiscard]] std::string node_label(NodeId node);
[[nodiscard]] std::string value_label(ValueId value);

[[nodiscard]] Result<ValueId> replay_operation(
    GraphBuilder& builder, const Node& source,
    std::span<const ValueId> operands);

[[nodiscard]] std::optional<Diagnostic> validate_replayed_node(
    const Node& source, const Node& result,
    std::span<const std::optional<ValueId>> value_map);

void append_provenance_entry(std::string& destination,
                             const NodeProvenance& entry);

[[nodiscard]] std::optional<Diagnostic> validate_source_provenance_domain(
    const VerifiedGraph& source, const GraphProvenance& provenance);

}  // namespace tensorkiln::detail
