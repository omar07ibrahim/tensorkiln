#include "tensorkiln/graph.hpp"

#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "float_bits.hpp"
#include "tensorkiln/shape_inference.hpp"

namespace tensorkiln {
namespace {

inline constexpr std::uint64_t kFnvOffsetBasis = UINT64_C(14695981039346656037);
inline constexpr std::uint64_t kFnvPrime = UINT64_C(1099511628211);

[[nodiscard]] std::uint64_t next_owner() noexcept {
  static std::atomic<std::uint64_t> next{1U};
  const std::uint64_t owner = next.fetch_add(1U, std::memory_order_relaxed);
  if (owner == 0U) {
    std::terminate();
  }
  return owner;
}

[[nodiscard]] bool ascii_alpha(const char character) noexcept {
  return (character >= 'a' && character <= 'z') ||
         (character >= 'A' && character <= 'Z');
}

[[nodiscard]] bool ascii_digit(const char character) noexcept {
  return character >= '0' && character <= '9';
}

[[nodiscard]] bool valid_name_character(const char character) noexcept {
  return ascii_alpha(character) || ascii_digit(character) ||
         character == '_' || character == '.' || character == '-';
}

[[nodiscard]] std::optional<Diagnostic> validate_name(
    const std::string& name, const std::string_view role,
    const std::size_t max_bytes) {
  if (name.size() > max_bytes) {
    return Diagnostic{
        ErrorCode::name_limit_exceeded,
        std::string(role) + " name uses " + std::to_string(name.size()) +
            " bytes; limit is " + std::to_string(max_bytes),
    };
  }
  if (name.empty()) {
    return Diagnostic{
        ErrorCode::invalid_name,
        std::string(role) + " name must not be empty",
    };
  }
  if (!ascii_alpha(name.front()) && name.front() != '_') {
    return Diagnostic{
        ErrorCode::invalid_name,
        std::string(role) + " name must begin with ASCII letter or underscore",
    };
  }
  for (const char character : name) {
    if (!valid_name_character(character)) {
      return Diagnostic{
          ErrorCode::invalid_name,
          std::string(role) + " name contains unsupported characters: " + name,
      };
    }
  }
  return std::nullopt;
}

[[nodiscard]] Diagnostic duplicate_name_error(const std::string_view role,
                                              const std::string& name) {
  return Diagnostic{
      ErrorCode::duplicate_name,
      std::string(role) + " name is already defined: " + name,
  };
}

[[nodiscard]] Diagnostic value_error(const ValueId value) {
  return Diagnostic{
      ErrorCode::value_not_found,
      "value %" + std::to_string(value.ordinal()) +
          " does not belong to this graph builder",
  };
}

[[nodiscard]] Diagnostic node_limit_error(const std::uint32_t limit) {
  return Diagnostic{
      ErrorCode::graph_node_limit_exceeded,
      "graph node limit " + std::to_string(limit) + " is exhausted",
  };
}

[[nodiscard]] Diagnostic output_limit_error(const std::uint32_t limit) {
  return Diagnostic{
      ErrorCode::graph_output_limit_exceeded,
      "graph output limit " + std::to_string(limit) + " is exhausted",
  };
}

[[nodiscard]] Diagnostic finished_error() {
  return Diagnostic{
      ErrorCode::builder_finished,
      "graph builder has already produced a verified graph",
  };
}

[[nodiscard]] std::uint64_t fingerprint(const std::span<const float> data) {
  std::uint64_t hash = kFnvOffsetBasis;
  for (const float& value : data) {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    for (std::uint32_t shift = 0U; shift < 32U; shift += 8U) {
      const auto byte = static_cast<std::uint8_t>((bits >> shift) & 0xffU);
      hash ^= byte;
      hash *= kFnvPrime;
    }
  }
  return hash;
}

[[nodiscard]] std::string hex64(std::uint64_t value) {
  constexpr std::string_view digits = "0123456789abcdef";
  std::string result(16U, '0');
  for (std::size_t index = result.size(); index > 0U; --index) {
    const auto digit = static_cast<std::size_t>(value & 0xfU);
    result[index - 1U] = digits[digit];
    value >>= 4U;
  }
  return result;
}

}  // namespace

ConstantOp::ConstantOp(std::string operation_name,
                       std::vector<float> operation_data,
                       const std::uint64_t operation_fingerprint)
    : name(std::move(operation_name)),
      data(std::move(operation_data)),
      fingerprint(operation_fingerprint) {}

ConstantOp::ConstantOp(const ConstantOp& other)
    : name(other.name),
      data(detail::copy_float_bits(other.data)),
      fingerprint(other.fingerprint) {}

ConstantOp& ConstantOp::operator=(const ConstantOp& other) {
  if (this == &other) {
    return *this;
  }
  std::string copied_name = other.name;
  std::vector<float> copied_data = detail::copy_float_bits(other.data);
  name = std::move(copied_name);
  data = std::move(copied_data);
  fingerprint = other.fingerprint;
  return *this;
}

Node::Node(const NodeId id, Operation operation, std::vector<ValueId> inputs,
           const ValueId output, TensorType output_type)
    : id_(id),
      operation_(std::move(operation)),
      inputs_(std::move(inputs)),
      output_(output),
      output_type_(std::move(output_type)) {}

GraphOutput::GraphOutput(const OutputId id, std::string name,
                         const ValueId value)
    : id_(id), name_(std::move(name)), value_(value) {}

VerifiedGraph::VerifiedGraph(const GraphLimits limits,
                             const std::uint64_t owner,
                             std::vector<Node> nodes,
                             std::vector<GraphOutput> outputs)
    : limits_(limits),
      owner_(owner),
      nodes_(std::move(nodes)),
      outputs_(std::move(outputs)) {}

const Node* VerifiedGraph::node(const NodeId id) const noexcept {
  if (id.owner_ != owner_) {
    return nullptr;
  }
  const std::size_t index = static_cast<std::size_t>(id.ordinal_);
  if (index >= nodes_.size()) {
    return nullptr;
  }
  return &nodes_[index];
}

const TensorType* VerifiedGraph::type(const ValueId value) const noexcept {
  if (value.owner_ != owner_) {
    return nullptr;
  }
  const Node* definition = node(NodeId{value.ordinal_, owner_});
  if (definition == nullptr) {
    return nullptr;
  }
  return &definition->output_type();
}

std::string VerifiedGraph::dump() const {
  std::string result{"tensorkiln.graph v0 {\n"};
  for (const Node& definition : nodes_) {
    result += "  #n" + std::to_string(definition.id().ordinal()) + " %" +
              std::to_string(definition.output().ordinal()) + " = ";

    const Operation& operation = definition.operation();
    if (const auto* input = std::get_if<InputOp>(&operation)) {
      result += "input @" + input->name;
    } else if (const auto* constant = std::get_if<ConstantOp>(&operation)) {
      result += "constant @" + constant->name + " {elements=" +
                std::to_string(constant->data.size()) +
                ", fnv1a64=0x" + hex64(constant->fingerprint) + "}";
    } else if (std::holds_alternative<AddOp>(operation)) {
      result += "add %" + std::to_string(definition.inputs()[0].ordinal()) +
                ", %" + std::to_string(definition.inputs()[1].ordinal());
    } else if (std::holds_alternative<MatMulOp>(operation)) {
      result += "matmul %" +
                std::to_string(definition.inputs()[0].ordinal()) + ", %" +
                std::to_string(definition.inputs()[1].ordinal());
    } else {
      result += "relu %" +
                std::to_string(definition.inputs()[0].ordinal());
    }
    result += " : " + definition.output_type().to_string() + "\n";
  }

  for (const GraphOutput& output : outputs_) {
    result += "  #o" + std::to_string(output.id().ordinal) + " output @" +
              output.name() + " = %" +
              std::to_string(output.value().ordinal()) + "\n";
  }
  result += "}\n";
  return result;
}

GraphBuilder::GraphBuilder(const GraphLimits limits)
    : limits_(limits), owner_(next_owner()) {}

const Node* GraphBuilder::find(const ValueId value) const noexcept {
  if (value.owner_ != owner_) {
    return nullptr;
  }
  const std::size_t index = static_cast<std::size_t>(value.ordinal_);
  if (index >= nodes_.size()) {
    return nullptr;
  }
  return &nodes_[index];
}

bool GraphBuilder::definition_name_exists(const std::string& name) const
    noexcept {
  for (const Node& definition : nodes_) {
    if (const auto* input = std::get_if<InputOp>(&definition.operation())) {
      if (input->name == name) {
        return true;
      }
    }
    if (const auto* constant =
            std::get_if<ConstantOp>(&definition.operation())) {
      if (constant->name == name) {
        return true;
      }
    }
  }
  return false;
}

bool GraphBuilder::output_name_exists(const std::string& name) const noexcept {
  for (const GraphOutput& output : outputs_) {
    if (output.name() == name) {
      return true;
    }
  }
  return false;
}

Result<TensorType> GraphBuilder::normalize_type(TensorType type) const {
  const auto shape = Shape::create(type.shape().extents(), limits_.shape_limits);
  if (!shape.has_value()) {
    return Result<TensorType>::failure(*shape.error_if());
  }
  return TensorType::create(*shape.value_if(), type.element_type(),
                            limits_.tensor_limits);
}

Result<ValueId> GraphBuilder::commit_node(Operation operation,
                                          std::vector<ValueId> inputs,
                                          TensorType output_type) {
  if (nodes_.size() >= static_cast<std::size_t>(limits_.max_nodes)) {
    return Result<ValueId>::failure(node_limit_error(limits_.max_nodes));
  }
  if (nodes_.size() >=
      static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
    return Result<ValueId>::failure(
        node_limit_error(std::numeric_limits<std::uint32_t>::max()));
  }

  nodes_.reserve(nodes_.size() + 1U);
  const auto ordinal = static_cast<std::uint32_t>(nodes_.size());
  const ValueId value{ordinal, owner_};
  nodes_.push_back(Node(NodeId{ordinal, owner_}, std::move(operation),
                        std::move(inputs), value, std::move(output_type)));
  return Result<ValueId>::success(value);
}

Result<ValueId> GraphBuilder::input(std::string name, TensorType type) {
  if (finished_) {
    return Result<ValueId>::failure(finished_error());
  }
  if (const auto error = validate_name(name, "definition", limits_.max_name_bytes)) {
    return Result<ValueId>::failure(*error);
  }
  if (definition_name_exists(name)) {
    return Result<ValueId>::failure(duplicate_name_error("definition", name));
  }
  auto normalized = normalize_type(std::move(type));
  if (!normalized.has_value()) {
    return Result<ValueId>::failure(*normalized.error_if());
  }
  return commit_node(InputOp{std::move(name)}, {},
                     std::move(*normalized.value_if()));
}

Result<ValueId> GraphBuilder::constant(std::string name, TensorType type,
                                       const std::span<const float> data) {
  if (finished_) {
    return Result<ValueId>::failure(finished_error());
  }
  if (const auto error = validate_name(name, "definition", limits_.max_name_bytes)) {
    return Result<ValueId>::failure(*error);
  }
  if (definition_name_exists(name)) {
    return Result<ValueId>::failure(duplicate_name_error("definition", name));
  }

  auto normalized = normalize_type(std::move(type));
  if (!normalized.has_value()) {
    return Result<ValueId>::failure(*normalized.error_if());
  }
  const std::uint64_t expected = normalized.value_if()->numel();
  if (expected != static_cast<std::uint64_t>(data.size())) {
    return Result<ValueId>::failure(Diagnostic{
        ErrorCode::constant_size_mismatch,
        "constant @" + name + " expects " + std::to_string(expected) +
            " values but received " + std::to_string(data.size()),
    });
  }
  if (expected > limits_.max_constant_elements - constant_elements_) {
    return Result<ValueId>::failure(Diagnostic{
        ErrorCode::constant_element_limit_exceeded,
        "constant data would exceed graph limit " +
            std::to_string(limits_.max_constant_elements) + " elements",
    });
  }
  if (nodes_.size() >= static_cast<std::size_t>(limits_.max_nodes)) {
    return Result<ValueId>::failure(node_limit_error(limits_.max_nodes));
  }

  std::vector<float> owned = detail::copy_float_bits(data);
  const std::uint64_t data_fingerprint = fingerprint(owned);
  auto committed = commit_node(
      ConstantOp{std::move(name), std::move(owned), data_fingerprint}, {},
      std::move(*normalized.value_if()));
  if (committed.has_value()) {
    constant_elements_ += expected;
  }
  return committed;
}

Result<ValueId> GraphBuilder::add(const ValueId left, const ValueId right) {
  if (finished_) {
    return Result<ValueId>::failure(finished_error());
  }
  const Node* left_node = find(left);
  if (left_node == nullptr) {
    return Result<ValueId>::failure(value_error(left));
  }
  const Node* right_node = find(right);
  if (right_node == nullptr) {
    return Result<ValueId>::failure(value_error(right));
  }

  auto shape = infer_broadcast_shape(left_node->output_type().shape(),
                                     right_node->output_type().shape(),
                                     limits_.shape_limits);
  if (!shape.has_value()) {
    return Result<ValueId>::failure(*shape.error_if());
  }
  auto type = TensorType::create(*shape.value_if(), ElementType::f32,
                                 limits_.tensor_limits);
  if (!type.has_value()) {
    return Result<ValueId>::failure(*type.error_if());
  }
  return commit_node(AddOp{}, {left, right}, std::move(*type.value_if()));
}

Result<ValueId> GraphBuilder::matmul(const ValueId left, const ValueId right) {
  if (finished_) {
    return Result<ValueId>::failure(finished_error());
  }
  const Node* left_node = find(left);
  if (left_node == nullptr) {
    return Result<ValueId>::failure(value_error(left));
  }
  const Node* right_node = find(right);
  if (right_node == nullptr) {
    return Result<ValueId>::failure(value_error(right));
  }

  auto shape = infer_matmul_shape(left_node->output_type().shape(),
                                  right_node->output_type().shape(),
                                  limits_.shape_limits);
  if (!shape.has_value()) {
    return Result<ValueId>::failure(*shape.error_if());
  }
  auto type = TensorType::create(*shape.value_if(), ElementType::f32,
                                 limits_.tensor_limits);
  if (!type.has_value()) {
    return Result<ValueId>::failure(*type.error_if());
  }
  return commit_node(MatMulOp{}, {left, right}, std::move(*type.value_if()));
}

Result<ValueId> GraphBuilder::relu(const ValueId input) {
  if (finished_) {
    return Result<ValueId>::failure(finished_error());
  }
  const Node* input_node = find(input);
  if (input_node == nullptr) {
    return Result<ValueId>::failure(value_error(input));
  }
  return commit_node(ReluOp{}, {input}, input_node->output_type());
}

Result<OutputId> GraphBuilder::output(std::string name, const ValueId value) {
  if (finished_) {
    return Result<OutputId>::failure(finished_error());
  }
  if (const auto error = validate_name(name, "output", limits_.max_name_bytes)) {
    return Result<OutputId>::failure(*error);
  }
  if (output_name_exists(name)) {
    return Result<OutputId>::failure(duplicate_name_error("output", name));
  }
  if (find(value) == nullptr) {
    return Result<OutputId>::failure(value_error(value));
  }
  if (outputs_.size() >= static_cast<std::size_t>(limits_.max_outputs)) {
    return Result<OutputId>::failure(output_limit_error(limits_.max_outputs));
  }

  outputs_.reserve(outputs_.size() + 1U);
  const auto ordinal = static_cast<std::uint32_t>(outputs_.size());
  const OutputId id{ordinal};
  outputs_.push_back(GraphOutput(id, std::move(name), value));
  return Result<OutputId>::success(id);
}

Result<VerifiedGraph> GraphBuilder::finish() && {
  if (finished_) {
    return Result<VerifiedGraph>::failure(finished_error());
  }
  if (outputs_.empty()) {
    return Result<VerifiedGraph>::failure(Diagnostic{
        ErrorCode::graph_has_no_outputs,
        "verified graph requires at least one declared output",
    });
  }

  finished_ = true;
  return Result<VerifiedGraph>::success(
      VerifiedGraph(limits_, owner_, std::move(nodes_), std::move(outputs_)));
}

}  // namespace tensorkiln
