#include "tensorkiln/provenance.hpp"

#include <algorithm>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace tensorkiln {
namespace {

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

}  // namespace

SourceDefinition::SourceDefinition(const NodeId node,
                                   const ValueId value) noexcept
    : node_(node), value_(value) {}

NodeProvenance::NodeProvenance(const NodeId result_node,
                               const ValueId result_value,
                               std::vector<SourceDefinition> sources)
    : result_node_(result_node),
      result_value_(result_value),
      sources_(std::move(sources)) {}

GraphProvenance::GraphProvenance(std::vector<NodeProvenance> entries)
    : entries_(std::move(entries)) {}

const NodeProvenance* GraphProvenance::for_result(
    const NodeId node) const noexcept {
  const std::size_t index = static_cast<std::size_t>(node.ordinal());
  if (index >= entries_.size() || entries_[index].result_node() != node) {
    return nullptr;
  }
  return &entries_[index];
}

const NodeProvenance* GraphProvenance::for_result(
    const ValueId value) const noexcept {
  const std::size_t index = static_cast<std::size_t>(value.ordinal());
  if (index >= entries_.size() || entries_[index].result_value() != value) {
    return nullptr;
  }
  return &entries_[index];
}

const NodeProvenance* GraphProvenance::for_source(
    const NodeId node) const noexcept {
  for (const NodeProvenance& entry : entries_) {
    for (const SourceDefinition& source : entry.sources()) {
      if (source.node() == node) {
        return &entry;
      }
    }
  }
  return nullptr;
}

const NodeProvenance* GraphProvenance::for_source(
    const ValueId value) const noexcept {
  for (const NodeProvenance& entry : entries_) {
    for (const SourceDefinition& source : entry.sources()) {
      if (source.value() == value) {
        return &entry;
      }
    }
  }
  return nullptr;
}

Result<GraphProvenance> GraphProvenance::compose(
    const GraphProvenance& upstream) const {
  std::vector<NodeProvenance> composed;
  composed.reserve(entries_.size());

  for (const NodeProvenance& entry : entries_) {
    std::vector<SourceDefinition> sources;
    for (const SourceDefinition& immediate_source : entry.sources()) {
      const NodeProvenance* by_node =
          upstream.for_result(immediate_source.node());
      const NodeProvenance* by_value =
          upstream.for_result(immediate_source.value());
      if (by_node == nullptr || by_value == nullptr || by_node != by_value) {
        return Result<GraphProvenance>::failure(provenance_error(
            "upstream provenance does not describe " +
            node_label(immediate_source.node()) + " " +
            value_label(immediate_source.value())));
      }
      sources.insert(sources.end(), by_node->sources().begin(),
                     by_node->sources().end());
    }

    std::stable_sort(
        sources.begin(), sources.end(),
        [](const SourceDefinition& left, const SourceDefinition& right) {
          return left.node().ordinal() < right.node().ordinal();
        });
    for (std::size_t index = 1U; index < sources.size(); ++index) {
      const SourceDefinition& previous = sources[index - 1U];
      const SourceDefinition& current = sources[index];
      if (previous.node().ordinal() == current.node().ordinal() &&
          previous != current) {
        return Result<GraphProvenance>::failure(provenance_error(
            "upstream provenance mixes source domains at ordinal " +
            std::to_string(current.node().ordinal())));
      }
    }
    sources.erase(std::unique(sources.begin(), sources.end()), sources.end());
    composed.push_back(NodeProvenance(
        entry.result_node(), entry.result_value(), std::move(sources)));
  }

  return Result<GraphProvenance>::success(
      GraphProvenance(std::move(composed)));
}

std::string GraphProvenance::dump() const {
  std::string result{"tensorkiln.provenance v0 {\n"};
  for (const NodeProvenance& entry : entries_) {
    append_provenance_entry(result, entry);
  }
  result += "}\n";
  return result;
}

}  // namespace tensorkiln
