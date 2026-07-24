#include "tensorkiln/execution_plan.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "compiler_support.hpp"
#include "execution_plan_support.hpp"

namespace tensorkiln {
namespace {

[[nodiscard]] bool valid_kernel_kind(const DenseKernelKind kernel) noexcept {
  switch (kernel) {
    case DenseKernelKind::add_contiguous_f32:
    case DenseKernelKind::add_broadcast_f32:
    case DenseKernelKind::matmul_rank2_f32:
    case DenseKernelKind::matmul_batched_f32:
    case DenseKernelKind::relu_contiguous_f32:
      return true;
  }
  return false;
}

class ReverseKernelClassifier final {
 public:
  ReverseKernelClassifier(const VerifiedGraph& graph, const Node& node)
      : graph_(graph), node_(node) {}

  [[nodiscard]] std::optional<DenseKernelKind> operator()(
      const InputOp&) const noexcept {
    return std::nullopt;
  }
  [[nodiscard]] std::optional<DenseKernelKind> operator()(
      const ConstantOp&) const noexcept {
    return std::nullopt;
  }
  [[nodiscard]] std::optional<DenseKernelKind> operator()(
      const AddOp&) const noexcept {
    if (node_.inputs().size() != 2U) {
      return std::nullopt;
    }
    const TensorType* left = graph_.type(node_.inputs()[0]);
    const TensorType* right = graph_.type(node_.inputs()[1]);
    if (left == nullptr || right == nullptr) {
      return std::nullopt;
    }
    if (left->shape() == node_.output_type().shape() &&
        right->shape() == node_.output_type().shape()) {
      return DenseKernelKind::add_contiguous_f32;
    }
    return DenseKernelKind::add_broadcast_f32;
  }
  [[nodiscard]] std::optional<DenseKernelKind> operator()(
      const MatMulOp&) const noexcept {
    if (node_.inputs().size() != 2U) {
      return std::nullopt;
    }
    const TensorType* left = graph_.type(node_.inputs()[0]);
    const TensorType* right = graph_.type(node_.inputs()[1]);
    if (left == nullptr || right == nullptr) {
      return std::nullopt;
    }
    if (left->shape().rank() == 2U && right->shape().rank() == 2U &&
        node_.output_type().shape().rank() == 2U) {
      return DenseKernelKind::matmul_rank2_f32;
    }
    return DenseKernelKind::matmul_batched_f32;
  }
  [[nodiscard]] std::optional<DenseKernelKind> operator()(
      const ReluOp&) const noexcept {
    if (node_.inputs().size() != 1U) {
      return std::nullopt;
    }
    return DenseKernelKind::relu_contiguous_f32;
  }

 private:
  const VerifiedGraph& graph_;
  const Node& node_;
};

[[nodiscard]] Result<ExecutionPlan> verifier_invariant(std::string message) {
  return Result<ExecutionPlan>::failure(
      detail::invariant_error(std::move(message)));
}

struct ReverseStepEvidence final {
  std::uint32_t source_node_ordinal;
  DenseKernelKind kernel;
  std::uint64_t scalar_steps;
};

[[nodiscard]] Result<std::uint64_t> reverse_scalar_work(
    const VerifiedGraph& graph, const Node& node) {
  if (std::holds_alternative<AddOp>(node.operation()) ||
      std::holds_alternative<ReluOp>(node.operation())) {
    return Result<std::uint64_t>::success(node.output_type().numel());
  }
  if (!std::holds_alternative<MatMulOp>(node.operation()) ||
      node.inputs().size() != 2U) {
    return Result<std::uint64_t>::failure(detail::invariant_error(
        "plan verifier cannot derive scalar work for " +
        detail::node_label(node.id())));
  }
  const TensorType* left = graph.type(node.inputs()[0]);
  if (left == nullptr || left->shape().rank() < 2U) {
    return Result<std::uint64_t>::failure(detail::invariant_error(
        "plan verifier found an invalid matmul left operand at " +
        detail::node_label(node.id())));
  }
  const std::int64_t signed_inner =
      left->shape().extents()[left->shape().rank() - 1U];
  if (signed_inner <= 0) {
    return Result<std::uint64_t>::failure(detail::invariant_error(
        "plan verifier found a non-positive matmul reduction extent at " +
        detail::node_label(node.id())));
  }
  const auto inner = static_cast<std::uint64_t>(signed_inner);
  if (node.output_type().numel() >
      std::numeric_limits<std::uint64_t>::max() / inner) {
    return Result<std::uint64_t>::failure(detail::execution_plan_error(
        ErrorCode::plan_work_overflow,
        "scalar work overflows uint64 at " +
            detail::node_label(node.id())));
  }
  return Result<std::uint64_t>::success(node.output_type().numel() * inner);
}

}  // namespace

Result<ExecutionPlan> ExecutionPlanVerifier::verify(
    const VerifiedGraph& source, const ExecutionPlanCandidate& candidate,
    const ExecutionPlanLimits limits) {
  auto analyzed = detail::analyze_execution_plan_source(source, limits);
  if (!analyzed.has_value()) {
    return Result<ExecutionPlan>::failure(*analyzed.error_if());
  }
  detail::ExecutionPlanSourceAnalysis analysis =
      std::move(*analyzed.value_if());

  std::vector<ReverseStepEvidence> expected_steps;
  expected_steps.reserve(static_cast<std::size_t>(analysis.step_count));
  std::uint64_t reverse_scalar_total = 0U;
  for (auto node_iterator = source.nodes().rbegin();
       node_iterator != source.nodes().rend(); ++node_iterator) {
    const Node& node = *node_iterator;
    if (std::holds_alternative<InputOp>(node.operation()) ||
        std::holds_alternative<ConstantOp>(node.operation())) {
      continue;
    }
    const std::optional<DenseKernelKind> kernel =
        std::visit(ReverseKernelClassifier(source, node), node.operation());
    if (!kernel.has_value()) {
      return verifier_invariant(
          "plan verifier could not classify materializing " +
          detail::node_label(node.id()));
    }
    auto scalar_work = reverse_scalar_work(source, node);
    if (!scalar_work.has_value()) {
      return Result<ExecutionPlan>::failure(*scalar_work.error_if());
    }
    if (*scalar_work.value_if() >
        std::numeric_limits<std::uint64_t>::max() - reverse_scalar_total) {
      return Result<ExecutionPlan>::failure(detail::execution_plan_error(
          ErrorCode::plan_work_overflow,
          "scalar work total overflows uint64 at " +
              detail::node_label(node.id())));
    }
    reverse_scalar_total += *scalar_work.value_if();
    expected_steps.push_back(ReverseStepEvidence{
        node.id().ordinal(),
        *kernel,
        *scalar_work.value_if(),
    });
  }
  std::reverse(expected_steps.begin(), expected_steps.end());
  if (expected_steps.size() !=
          static_cast<std::size_t>(analysis.step_count) ||
      reverse_scalar_total != analysis.scalar_steps) {
    return verifier_invariant(
        "plan preflight and reverse verifier derived different execution work");
  }

  if (candidate.steps.size() !=
      static_cast<std::size_t>(analysis.step_count)) {
    return Result<ExecutionPlan>::failure(detail::execution_plan_error(
        ErrorCode::plan_step_count_mismatch,
        "candidate has " + std::to_string(candidate.steps.size()) +
            " step specs; expected " +
            std::to_string(analysis.step_count)));
  }

  const std::span<const Node> source_nodes = source.nodes();
  for (std::size_t step_index = 0U; step_index < candidate.steps.size();
       ++step_index) {
    const ExecutionStepSpec& spec = candidate.steps[step_index];
    const std::uint32_t expected_node =
        expected_steps[step_index].source_node_ordinal;
    if (spec.source_node_ordinal != expected_node) {
      return Result<ExecutionPlan>::failure(detail::execution_plan_error(
          ErrorCode::plan_step_source_mismatch,
          "candidate step @" + std::to_string(step_index) + " names #n" +
              std::to_string(spec.source_node_ordinal) + "; expected #n" +
              std::to_string(expected_node)));
    }
    if (!valid_kernel_kind(spec.kernel)) {
      return Result<ExecutionPlan>::failure(detail::execution_plan_error(
          ErrorCode::plan_kernel_invalid,
          "candidate step @" + std::to_string(step_index) +
              " has invalid kernel value " +
              std::to_string(static_cast<unsigned int>(spec.kernel))));
    }
    const Node& node = source_nodes[static_cast<std::size_t>(expected_node)];
    const DenseKernelKind expected = expected_steps[step_index].kernel;
    if (spec.kernel != expected) {
      return Result<ExecutionPlan>::failure(detail::execution_plan_error(
          ErrorCode::plan_kernel_incompatible,
          "candidate kernel " +
              std::string(dense_kernel_kind_name(spec.kernel)) +
              " is incompatible with " + detail::node_label(node.id()) +
              "; expected " +
              std::string(dense_kernel_kind_name(expected))));
    }
  }

  VerifiedGraph owned_graph = source;
  auto arena_verified = GraphArenaPlacementVerifier::verify(
      owned_graph, candidate.placements, limits.arena_limits);
  if (!arena_verified.has_value()) {
    return Result<ExecutionPlan>::failure(*arena_verified.error_if());
  }
  GraphArenaLoweringResult arena =
      std::move(*arena_verified.value_if());
  if (arena.source_node_count() != analysis.value_count ||
      arena.execution_step_count() != analysis.step_count ||
      arena.arena_plan().limits() != limits.arena_limits ||
      arena.values_by_buffer_ordinal().size() !=
          static_cast<std::size_t>(analysis.step_count)) {
    return verifier_invariant(
        "plan verifier and arena verifier reconstructed different domains");
  }
  for (std::size_t step_index = 0U; step_index < expected_steps.size();
       ++step_index) {
    const ValueId* arena_value =
        arena.value_at(static_cast<std::uint32_t>(step_index));
    const Node& expected_node = owned_graph.nodes()[static_cast<std::size_t>(
        expected_steps[step_index].source_node_ordinal)];
    if (arena_value == nullptr || *arena_value != expected_node.output()) {
      return verifier_invariant(
          "plan verifier and arena verifier derived different step order at @" +
          std::to_string(step_index));
    }
  }

  std::vector<PlanValue> values;
  values.reserve(static_cast<std::size_t>(analysis.value_count));
  std::uint32_t input_ordinal = 0U;
  std::uint32_t constant_ordinal = 0U;
  for (const Node& node : owned_graph.nodes()) {
    auto prepared_layout =
        detail::prepare_dense_layout(node.output_type(), node.id());
    if (!prepared_layout.has_value()) {
      return Result<ExecutionPlan>::failure(*prepared_layout.error_if());
    }
    const detail::PreparedDenseLayout& raw_layout =
        *prepared_layout.value_if();
    const DenseLayout layout(raw_layout.strides_elements, raw_layout.rank,
                             raw_layout.elements);

    PlanStorage storage(PlanStorageKind::input, 0U, 0U);
    if (std::holds_alternative<InputOp>(node.operation())) {
      storage = PlanStorage(PlanStorageKind::input, input_ordinal, 0U);
      ++input_ordinal;
    } else if (std::holds_alternative<ConstantOp>(node.operation())) {
      storage =
          PlanStorage(PlanStorageKind::constant, constant_ordinal, 0U);
      ++constant_ordinal;
    } else {
      const std::optional<std::uint32_t> buffer =
          arena.buffer_ordinal(node.output());
      const ArenaAllocation* allocation =
          arena.allocation_for(node.output());
      if (!buffer.has_value() || allocation == nullptr ||
          allocation->buffer_ordinal() != *buffer) {
        return verifier_invariant(
            "arena mapping omitted materialized " +
            detail::value_label(node.output()));
      }
      storage = PlanStorage(PlanStorageKind::arena, *buffer,
                            allocation->offset_bytes());
    }
    values.push_back(
        PlanValue(node.output(), node.output_type(), layout, storage));
  }
  if (input_ordinal != analysis.input_count ||
      constant_ordinal != analysis.constant_count) {
    return verifier_invariant(
        "plan verifier external storage counts changed during construction");
  }

  std::vector<ExecutionStep> steps;
  steps.reserve(static_cast<std::size_t>(analysis.step_count));
  for (std::size_t step_index = 0U; step_index < candidate.steps.size();
       ++step_index) {
    const ExecutionStepSpec& spec = candidate.steps[step_index];
    const Node& node = owned_graph.nodes()[static_cast<std::size_t>(
        spec.source_node_ordinal)];
    std::vector<ValueId> operands(node.inputs().begin(), node.inputs().end());
    steps.push_back(ExecutionStep(
        static_cast<std::uint32_t>(step_index), node.id(), spec.kernel,
        std::move(operands), node.output(),
        expected_steps[step_index].scalar_steps));
  }

  const ExecutionPlanStats stats{
      analysis.value_count,
      analysis.input_count,
      analysis.constant_count,
      analysis.step_count,
      analysis.output_count,
      analysis.owned_constant_bytes,
      analysis.scalar_steps,
      arena.arena_plan().workspace_bytes(),
  };
  return Result<ExecutionPlan>::success(ExecutionPlan(
      limits, stats, std::move(owned_graph), std::move(arena),
      std::move(values), std::move(steps)));
}

}  // namespace tensorkiln
