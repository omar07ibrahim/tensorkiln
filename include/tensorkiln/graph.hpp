#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <variant>
#include <vector>

#include "tensorkiln/tensor_type.hpp"

namespace tensorkiln {

class GraphBuilder;
class VerifiedGraph;

class ValueId final {
 public:
  [[nodiscard]] std::uint32_t ordinal() const noexcept { return ordinal_; }

  friend bool operator==(const ValueId&, const ValueId&) noexcept = default;

 private:
  friend class GraphBuilder;
  friend class VerifiedGraph;

  ValueId(std::uint32_t ordinal, std::uint64_t owner) noexcept
      : ordinal_(ordinal), owner_(owner) {}

  std::uint32_t ordinal_;
  std::uint64_t owner_;
};

class NodeId final {
 public:
  [[nodiscard]] std::uint32_t ordinal() const noexcept { return ordinal_; }

  friend bool operator==(const NodeId&, const NodeId&) noexcept = default;

 private:
  friend class GraphBuilder;
  friend class VerifiedGraph;

  NodeId(std::uint32_t ordinal, std::uint64_t owner) noexcept
      : ordinal_(ordinal), owner_(owner) {}

  std::uint32_t ordinal_;
  std::uint64_t owner_;
};

struct OutputId final {
  std::uint32_t ordinal;

  friend bool operator==(const OutputId&, const OutputId&) noexcept = default;
};

struct InputOp final {
  std::string name;
};

struct ConstantOp final {
  ConstantOp(std::string name, std::vector<float> data,
             std::uint64_t fingerprint);
  ConstantOp(const ConstantOp& other);
  ConstantOp(ConstantOp&&) noexcept = default;
  ConstantOp& operator=(const ConstantOp& other);
  ConstantOp& operator=(ConstantOp&&) noexcept = default;
  ~ConstantOp() = default;

  std::string name;
  std::vector<float> data;
  std::uint64_t fingerprint;
};

struct AddOp final {};
struct MatMulOp final {};
struct ReluOp final {};

using Operation =
    std::variant<InputOp, ConstantOp, AddOp, MatMulOp, ReluOp>;

class Node final {
 public:
  Node(const Node&) = default;
  Node(Node&&) = default;
  Node& operator=(const Node&) = default;
  Node& operator=(Node&&) = default;
  ~Node() = default;

  [[nodiscard]] NodeId id() const noexcept { return id_; }
  [[nodiscard]] const Operation& operation() const noexcept {
    return operation_;
  }
  [[nodiscard]] std::span<const ValueId> inputs() const noexcept {
    return inputs_;
  }
  [[nodiscard]] ValueId output() const noexcept { return output_; }
  [[nodiscard]] const TensorType& output_type() const noexcept {
    return output_type_;
  }

 private:
  friend class GraphBuilder;

  Node(NodeId id, Operation operation, std::vector<ValueId> inputs,
       ValueId output, TensorType output_type);

  NodeId id_;
  Operation operation_;
  std::vector<ValueId> inputs_;
  ValueId output_;
  TensorType output_type_;
};

class GraphOutput final {
 public:
  GraphOutput(const GraphOutput&) = default;
  GraphOutput(GraphOutput&&) = default;
  GraphOutput& operator=(const GraphOutput&) = default;
  GraphOutput& operator=(GraphOutput&&) = default;
  ~GraphOutput() = default;

  [[nodiscard]] OutputId id() const noexcept { return id_; }
  [[nodiscard]] const std::string& name() const noexcept { return name_; }
  [[nodiscard]] ValueId value() const noexcept { return value_; }

 private:
  friend class GraphBuilder;

  GraphOutput(OutputId id, std::string name, ValueId value);

  OutputId id_;
  std::string name_;
  ValueId value_;
};

struct GraphLimits final {
  std::uint32_t max_nodes = 4096U;
  std::uint32_t max_outputs = 64U;
  std::size_t max_name_bytes = 128U;
  std::uint64_t max_constant_elements = UINT64_C(1) << 24U;
  ShapeLimits shape_limits{};
  TensorLimits tensor_limits{};

  friend bool operator==(const GraphLimits&, const GraphLimits&) noexcept =
      default;
};

class VerifiedGraph final {
 public:
  VerifiedGraph(const VerifiedGraph&) = default;
  VerifiedGraph(VerifiedGraph&&) = default;
  VerifiedGraph& operator=(const VerifiedGraph&) = default;
  VerifiedGraph& operator=(VerifiedGraph&&) = default;
  ~VerifiedGraph() = default;

  [[nodiscard]] std::span<const Node> nodes() const noexcept { return nodes_; }
  [[nodiscard]] std::span<const GraphOutput> outputs() const noexcept {
    return outputs_;
  }
  [[nodiscard]] const GraphLimits& limits() const noexcept { return limits_; }
  [[nodiscard]] const Node* node(NodeId id) const noexcept;
  [[nodiscard]] const TensorType* type(ValueId value) const noexcept;
  [[nodiscard]] std::string dump() const;

 private:
  friend class GraphBuilder;

  VerifiedGraph(GraphLimits limits, std::uint64_t owner,
                std::vector<Node> nodes, std::vector<GraphOutput> outputs);

  GraphLimits limits_;
  std::uint64_t owner_;
  std::vector<Node> nodes_;
  std::vector<GraphOutput> outputs_;
};

class GraphBuilder final {
 public:
  explicit GraphBuilder(GraphLimits limits = GraphLimits{});

  GraphBuilder(const GraphBuilder&) = delete;
  GraphBuilder(GraphBuilder&&) = delete;
  GraphBuilder& operator=(const GraphBuilder&) = delete;
  GraphBuilder& operator=(GraphBuilder&&) = delete;
  ~GraphBuilder() = default;

  [[nodiscard]] Result<ValueId> input(std::string name, TensorType type);

  [[nodiscard]] Result<ValueId> constant(std::string name, TensorType type,
                                         std::span<const float> data);

  [[nodiscard]] Result<ValueId> add(ValueId left, ValueId right);
  [[nodiscard]] Result<ValueId> matmul(ValueId left, ValueId right);
  [[nodiscard]] Result<ValueId> relu(ValueId input);

  [[nodiscard]] Result<OutputId> output(std::string name, ValueId value);
  [[nodiscard]] Result<VerifiedGraph> finish() &&;

  [[nodiscard]] std::size_t node_count() const noexcept {
    return nodes_.size();
  }
  [[nodiscard]] std::size_t output_count() const noexcept {
    return outputs_.size();
  }
  [[nodiscard]] std::uint64_t constant_elements() const noexcept {
    return constant_elements_;
  }

 private:
  [[nodiscard]] const Node* find(ValueId value) const noexcept;
  [[nodiscard]] bool definition_name_exists(
      const std::string& name) const noexcept;
  [[nodiscard]] bool output_name_exists(const std::string& name) const noexcept;
  [[nodiscard]] Result<TensorType> normalize_type(TensorType type) const;
  [[nodiscard]] Result<ValueId> commit_node(
      Operation operation, std::vector<ValueId> inputs, TensorType output_type);

  GraphLimits limits_;
  std::uint64_t owner_;
  std::vector<Node> nodes_;
  std::vector<GraphOutput> outputs_;
  std::uint64_t constant_elements_ = 0U;
  bool finished_ = false;
};

}  // namespace tensorkiln
