#include "execution_plan_support.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <variant>

#include "compiler_support.hpp"

namespace tensorkiln::detail {
namespace {

class PlanOperationClass final {
 public:
  enum class Kind : std::uint8_t {
    input,
    constant,
    compute,
  };

  [[nodiscard]] Kind operator()(const InputOp&) const noexcept {
    return Kind::input;
  }
  [[nodiscard]] Kind operator()(const ConstantOp&) const noexcept {
    return Kind::constant;
  }
  [[nodiscard]] Kind operator()(const AddOp&) const noexcept {
    return Kind::compute;
  }
  [[nodiscard]] Kind operator()(const MatMulOp&) const noexcept {
    return Kind::compute;
  }
  [[nodiscard]] Kind operator()(const ReluOp&) const noexcept {
    return Kind::compute;
  }
};

[[nodiscard]] Result<ExecutionPlanSourceAnalysis> analysis_invariant(
    std::string message) {
  return Result<ExecutionPlanSourceAnalysis>::failure(
      invariant_error(std::move(message)));
}

[[nodiscard]] bool add_overflows(const std::uint64_t left,
                                 const std::uint64_t right) noexcept {
  return right > std::numeric_limits<std::uint64_t>::max() - left;
}

[[nodiscard]] Result<std::uint64_t> scalar_work_for(
    const VerifiedGraph& source, const Node& node) {
  if (std::holds_alternative<AddOp>(node.operation()) ||
      std::holds_alternative<ReluOp>(node.operation())) {
    return Result<std::uint64_t>::success(node.output_type().numel());
  }
  if (std::holds_alternative<MatMulOp>(node.operation())) {
    if (node.inputs().size() != 2U) {
      return Result<std::uint64_t>::failure(invariant_error(
          "plan matmul " + node_label(node.id()) + " does not have two operands"));
    }
    const TensorType* left = source.type(node.inputs()[0]);
    if (left == nullptr || left->shape().rank() < 2U) {
      return Result<std::uint64_t>::failure(invariant_error(
          "plan matmul " + node_label(node.id()) +
          " has an invalid left operand type"));
    }
    const std::int64_t signed_k =
        left->shape().extents()[left->shape().rank() - 1U];
    if (signed_k <= 0) {
      return Result<std::uint64_t>::failure(invariant_error(
          "plan matmul " + node_label(node.id()) +
          " has a non-positive reduction extent"));
    }
    const auto k = static_cast<std::uint64_t>(signed_k);
    if (node.output_type().numel() >
        std::numeric_limits<std::uint64_t>::max() / k) {
      return Result<std::uint64_t>::failure(execution_plan_error(
          ErrorCode::plan_work_overflow,
          "scalar work overflows uint64 at " + node_label(node.id())));
    }
    return Result<std::uint64_t>::success(node.output_type().numel() * k);
  }
  return Result<std::uint64_t>::failure(invariant_error(
      "requested scalar work for non-compute " + node_label(node.id())));
}

}  // namespace

Diagnostic execution_plan_error(const ErrorCode code, std::string message) {
  return Diagnostic{code, std::move(message)};
}

Result<ExecutionPlanSourceAnalysis> analyze_execution_plan_source(
    const VerifiedGraph& source, const ExecutionPlanLimits& limits) {
  const std::span<const Node> nodes = source.nodes();
  const std::span<const GraphOutput> outputs = source.outputs();
  if (nodes.size() > static_cast<std::size_t>(
                         std::numeric_limits<std::uint32_t>::max()) ||
      outputs.size() > static_cast<std::size_t>(
                           std::numeric_limits<std::uint32_t>::max())) {
    return analysis_invariant(
        "execution plan source counts exceed the uint32 domain");
  }
  if (outputs.empty()) {
    return analysis_invariant(
        "execution plan source has no graph outputs");
  }
  if (nodes.size() > static_cast<std::size_t>(limits.max_values)) {
    return Result<ExecutionPlanSourceAnalysis>::failure(execution_plan_error(
        ErrorCode::plan_value_limit_exceeded,
        "plan has " + std::to_string(nodes.size()) +
            " values; limit is " + std::to_string(limits.max_values)));
  }

  ExecutionPlanSourceAnalysis analysis{
      static_cast<std::uint32_t>(nodes.size()),
      0U,
      0U,
      0U,
      static_cast<std::uint32_t>(outputs.size()),
      0U,
      0U,
  };

  for (std::size_t source_index = 0U; source_index < nodes.size();
       ++source_index) {
    const Node& node = nodes[source_index];
    if (node.id().ordinal() != source_index ||
        node.output().ordinal() != source_index ||
        source.node(node.id()) != &node ||
        source.type(node.output()) != &node.output_type()) {
      return analysis_invariant(
          "execution plan source has a non-dense or foreign definition at index " +
          std::to_string(source_index));
    }
    for (const ValueId operand : node.inputs()) {
      const std::size_t operand_index =
          static_cast<std::size_t>(operand.ordinal());
      if (operand_index >= source_index || operand_index >= nodes.size() ||
          source.type(operand) == nullptr ||
          nodes[operand_index].output() != operand) {
        return analysis_invariant(
            "execution plan source has a foreign or forward operand " +
            value_label(operand) + " at " + node_label(node.id()));
      }
    }

    switch (std::visit(PlanOperationClass{}, node.operation())) {
      case PlanOperationClass::Kind::input:
        ++analysis.input_count;
        break;
      case PlanOperationClass::Kind::constant: {
        ++analysis.constant_count;
        const ConstantOp& constant = std::get<ConstantOp>(node.operation());
        if (constant.data.size() >
            static_cast<std::size_t>(
                std::numeric_limits<std::uint64_t>::max() / sizeof(float))) {
          return Result<ExecutionPlanSourceAnalysis>::failure(
              execution_plan_error(
                  ErrorCode::plan_constant_byte_limit_exceeded,
                  "owned constant bytes exceed the uint64 domain"));
        }
        const std::uint64_t bytes =
            static_cast<std::uint64_t>(constant.data.size()) * sizeof(float);
        if (add_overflows(analysis.owned_constant_bytes, bytes)) {
          return Result<ExecutionPlanSourceAnalysis>::failure(
              execution_plan_error(
                  ErrorCode::plan_constant_byte_limit_exceeded,
                  "owned constant byte total exceeds the uint64 domain"));
        }
        analysis.owned_constant_bytes += bytes;
        break;
      }
      case PlanOperationClass::Kind::compute:
        ++analysis.step_count;
        break;
    }
  }

  for (std::size_t output_index = 0U; output_index < outputs.size();
       ++output_index) {
    const GraphOutput& output = outputs[output_index];
    const std::size_t value_index =
        static_cast<std::size_t>(output.value().ordinal());
    if (output.id().ordinal != output_index || value_index >= nodes.size() ||
        source.type(output.value()) == nullptr ||
        nodes[value_index].output() != output.value()) {
      return analysis_invariant(
          "execution plan source has an invalid output at index " +
          std::to_string(output_index));
    }
  }

  if (analysis.step_count > limits.max_steps) {
    return Result<ExecutionPlanSourceAnalysis>::failure(execution_plan_error(
        ErrorCode::plan_step_limit_exceeded,
        "plan has " + std::to_string(analysis.step_count) +
            " steps; limit is " + std::to_string(limits.max_steps)));
  }
  if (analysis.output_count > limits.max_outputs) {
    return Result<ExecutionPlanSourceAnalysis>::failure(execution_plan_error(
        ErrorCode::plan_output_limit_exceeded,
        "plan has " + std::to_string(analysis.output_count) +
            " outputs; limit is " + std::to_string(limits.max_outputs)));
  }
  if (analysis.owned_constant_bytes > limits.max_owned_constant_bytes) {
    return Result<ExecutionPlanSourceAnalysis>::failure(execution_plan_error(
        ErrorCode::plan_constant_byte_limit_exceeded,
        "plan owns " + std::to_string(analysis.owned_constant_bytes) +
            " constant bytes; limit is " +
            std::to_string(limits.max_owned_constant_bytes)));
  }

  for (const Node& node : nodes) {
    if (std::visit(PlanOperationClass{}, node.operation()) !=
        PlanOperationClass::Kind::compute) {
      continue;
    }
    auto work = scalar_work_for(source, node);
    if (!work.has_value()) {
      return Result<ExecutionPlanSourceAnalysis>::failure(*work.error_if());
    }
    const std::uint64_t step_work = *work.value_if();
    if (add_overflows(analysis.scalar_steps, step_work)) {
      return Result<ExecutionPlanSourceAnalysis>::failure(
          execution_plan_error(
              ErrorCode::plan_work_overflow,
              "scalar work total overflows uint64 at " +
                  node_label(node.id())));
    }
    analysis.scalar_steps += step_work;
  }
  if (analysis.scalar_steps > limits.max_scalar_steps) {
    return Result<ExecutionPlanSourceAnalysis>::failure(execution_plan_error(
        ErrorCode::plan_scalar_step_limit_exceeded,
        "plan requires " + std::to_string(analysis.scalar_steps) +
            " scalar steps; limit is " +
            std::to_string(limits.max_scalar_steps)));
  }
  return Result<ExecutionPlanSourceAnalysis>::success(std::move(analysis));
}

Result<PreparedDenseLayout> prepare_dense_layout(
    const TensorType& type, const NodeId source_node) {
  const Shape& shape = type.shape();
  if (shape.rank() > kMaxRank) {
    return Result<PreparedDenseLayout>::failure(invariant_error(
        "plan layout rank exceeds kMaxRank at " + node_label(source_node)));
  }
  PreparedDenseLayout layout{{}, shape.rank(), shape.numel()};
  std::uint64_t stride = 1U;
  for (std::size_t axis = shape.rank(); axis > 0U; --axis) {
    const std::size_t current_axis = axis - 1U;
    const std::int64_t signed_extent = shape.extents()[current_axis];
    if (signed_extent <= 0) {
      return Result<PreparedDenseLayout>::failure(invariant_error(
          "plan layout has a non-positive extent at " +
          node_label(source_node)));
    }
    layout.strides_elements[current_axis] = stride;
    const auto extent = static_cast<std::uint64_t>(signed_extent);
    if (stride > std::numeric_limits<std::uint64_t>::max() / extent) {
      return Result<PreparedDenseLayout>::failure(invariant_error(
          "plan dense layout overflows uint64 at " +
          node_label(source_node)));
    }
    stride *= extent;
  }
  if (stride != shape.numel()) {
    return Result<PreparedDenseLayout>::failure(invariant_error(
        "plan dense layout element count disagrees at " +
        node_label(source_node)));
  }
  return Result<PreparedDenseLayout>::success(layout);
}

}  // namespace tensorkiln::detail
