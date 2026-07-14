#pragma once

#include <span>
#include <string>
#include <vector>

#include "tensorkiln/graph.hpp"

namespace tensorkiln {

class SourceDefinition final {
 public:
  [[nodiscard]] NodeId node() const noexcept { return node_; }
  [[nodiscard]] ValueId value() const noexcept { return value_; }

 friend bool operator==(const SourceDefinition&,
                         const SourceDefinition&) noexcept = default;

 private:
  friend class GraphProvenance;

  SourceDefinition(NodeId node, ValueId value) noexcept;

  NodeId node_;
  ValueId value_;
};

class NodeProvenance final {
 public:
  [[nodiscard]] NodeId result_node() const noexcept { return result_node_; }
  [[nodiscard]] ValueId result_value() const noexcept {
    return result_value_;
  }
  [[nodiscard]] std::span<const SourceDefinition> sources() const noexcept {
    return sources_;
  }

 private:
  friend class GraphProvenance;

  NodeProvenance(NodeId result_node, ValueId result_value,
                 std::vector<SourceDefinition> sources);

  NodeId result_node_;
  ValueId result_value_;
  std::vector<SourceDefinition> sources_;
};

class GraphProvenance final {
 public:
  GraphProvenance(const GraphProvenance&) = default;
  GraphProvenance(GraphProvenance&&) noexcept = default;
  GraphProvenance& operator=(const GraphProvenance&) = default;
  GraphProvenance& operator=(GraphProvenance&&) noexcept = default;
  ~GraphProvenance() = default;

  [[nodiscard]] static Result<GraphProvenance> create(
      const VerifiedGraph& result_graph, const VerifiedGraph& source_graph,
      std::vector<std::vector<NodeId>> source_nodes_by_result);

  [[nodiscard]] std::span<const NodeProvenance> entries() const noexcept {
    return entries_;
  }
  [[nodiscard]] const NodeProvenance* for_result(NodeId node) const noexcept;
  [[nodiscard]] const NodeProvenance* for_result(ValueId value) const noexcept;
  [[nodiscard]] const NodeProvenance* for_source(NodeId node) const noexcept;
  [[nodiscard]] const NodeProvenance* for_source(ValueId value) const noexcept;

  [[nodiscard]] Result<GraphProvenance> compose(
      const GraphProvenance& upstream) const;
  [[nodiscard]] std::string dump() const;

 private:
  explicit GraphProvenance(std::vector<NodeProvenance> entries);

  std::vector<NodeProvenance> entries_;
};

}  // namespace tensorkiln
