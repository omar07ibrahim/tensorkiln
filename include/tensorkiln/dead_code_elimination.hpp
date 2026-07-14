#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "tensorkiln/graph.hpp"

namespace tensorkiln {

struct DeadCodeEliminationStats final {
  std::uint32_t source_nodes;
  std::uint32_t retained_nodes;
  std::uint32_t removed_nodes;
  std::uint64_t source_constant_elements;
  std::uint64_t retained_constant_elements;
  std::uint64_t removed_constant_elements;

  friend bool operator==(const DeadCodeEliminationStats&,
                         const DeadCodeEliminationStats&) noexcept = default;
};

class SourceDefinition final {
 public:
  [[nodiscard]] NodeId node() const noexcept { return node_; }
  [[nodiscard]] ValueId value() const noexcept { return value_; }

  friend bool operator==(const SourceDefinition&,
                         const SourceDefinition&) noexcept = default;

 private:
  friend class GraphProvenance;
  friend class DeadCodeElimination;

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
  friend class DeadCodeElimination;

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
  friend class DeadCodeElimination;

  explicit GraphProvenance(std::vector<NodeProvenance> entries);

  std::vector<NodeProvenance> entries_;
};

class DeadCodeEliminationResult final {
 public:
  DeadCodeEliminationResult(const DeadCodeEliminationResult&) = delete;
  DeadCodeEliminationResult(DeadCodeEliminationResult&&) noexcept = default;
  DeadCodeEliminationResult& operator=(
      const DeadCodeEliminationResult&) = delete;
  DeadCodeEliminationResult& operator=(
      DeadCodeEliminationResult&&) noexcept = default;
  ~DeadCodeEliminationResult() = default;

  [[nodiscard]] const VerifiedGraph& graph() const noexcept { return graph_; }
  [[nodiscard]] const GraphProvenance& provenance() const noexcept {
    return provenance_;
  }
  [[nodiscard]] const DeadCodeEliminationStats& stats() const noexcept {
    return stats_;
  }
  [[nodiscard]] std::string dump() const;

 private:
  friend class DeadCodeElimination;

  DeadCodeEliminationResult(VerifiedGraph graph,
                            GraphProvenance provenance,
                            DeadCodeEliminationStats stats);

  VerifiedGraph graph_;
  GraphProvenance provenance_;
  DeadCodeEliminationStats stats_;
};

class DeadCodeElimination final {
 public:
  DeadCodeElimination() = delete;

  [[nodiscard]] static Result<DeadCodeEliminationResult> run(
      const VerifiedGraph& source);

  [[nodiscard]] static Result<DeadCodeEliminationResult> run(
      const VerifiedGraph& source,
      const GraphProvenance& source_provenance);
};

}  // namespace tensorkiln
