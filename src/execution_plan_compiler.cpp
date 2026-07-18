#include "tensorkiln/execution_plan.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "compiler_support.hpp"
#include "execution_plan_support.hpp"

namespace tensorkiln {
namespace {

class ForwardKernelSelector final {
 public:
  ForwardKernelSelector(const VerifiedGraph& graph, const Node& node)
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
    return left->shape() == node_.output_type().shape() &&
                   right->shape() == node_.output_type().shape()
               ? DenseKernelKind::add_contiguous_f32
               : DenseKernelKind::add_broadcast_f32;
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
    return left->shape().rank() == 2U && right->shape().rank() == 2U &&
                   node_.output_type().shape().rank() == 2U
               ? DenseKernelKind::matmul_rank2_f32
               : DenseKernelKind::matmul_batched_f32;
  }
  [[nodiscard]] std::optional<DenseKernelKind> operator()(
      const ReluOp&) const noexcept {
    return node_.inputs().size() == 1U
               ? std::optional<DenseKernelKind>(
                     DenseKernelKind::relu_contiguous_f32)
               : std::nullopt;
  }

 private:
  const VerifiedGraph& graph_;
  const Node& node_;
};

[[nodiscard]] Result<ExecutionPlan> compiler_invariant(std::string message) {
  return Result<ExecutionPlan>::failure(
      detail::invariant_error(std::move(message)));
}

}  // namespace

Result<ExecutionPlan> ExecutionPlanCompiler::run(
    const VerifiedGraph& source, const ExecutionPlanLimits limits) {
  auto analyzed = detail::analyze_execution_plan_source(source, limits);
  if (!analyzed.has_value()) {
    return Result<ExecutionPlan>::failure(*analyzed.error_if());
  }
  detail::ExecutionPlanSourceAnalysis analysis =
      std::move(*analyzed.value_if());

  auto lowered = GraphArenaLowering::run(source, limits.arena_limits);
  if (!lowered.has_value()) {
    return Result<ExecutionPlan>::failure(*lowered.error_if());
  }
  const GraphArenaLoweringResult& arena = *lowered.value_if();
  if (arena.source_node_count() != analysis.value_count ||
      arena.execution_step_count() != analysis.step_count) {
    return compiler_invariant(
        "execution plan compiler and arena lowering derived different domains");
  }

  std::vector<ExecutionStepSpec> specs;
  specs.reserve(static_cast<std::size_t>(analysis.step_count));
  for (const Node& node : source.nodes()) {
    if (std::holds_alternative<InputOp>(node.operation()) ||
        std::holds_alternative<ConstantOp>(node.operation())) {
      continue;
    }
    const std::optional<DenseKernelKind> kernel =
        std::visit(ForwardKernelSelector(source, node), node.operation());
    if (!kernel.has_value()) {
      return compiler_invariant(
          "execution plan compiler could not select a kernel for " +
          detail::node_label(node.id()));
    }
    specs.push_back(ExecutionStepSpec{node.id().ordinal(), *kernel});
  }
  if (specs.size() != static_cast<std::size_t>(analysis.step_count)) {
    return compiler_invariant(
        "execution plan compiler and preflight derived different step counts");
  }

  std::vector<ArenaPlacement> placements;
  placements.reserve(arena.arena_plan().allocations().size());
  for (const ArenaAllocation& allocation :
       arena.arena_plan().allocations()) {
    placements.push_back(ArenaPlacement{
        allocation.buffer_ordinal(),
        allocation.offset_bytes(),
    });
  }

  const ExecutionPlanCandidate candidate{specs, placements};
  auto verified = ExecutionPlanVerifier::verify(source, candidate, limits);
  if (!verified.has_value()) {
    return compiler_invariant(
        "execution plan compiler failed independent verification: " +
        std::string(error_code_name(verified.error_if()->code)) + ": " +
        verified.error_if()->message);
  }
  return verified;
}

}  // namespace tensorkiln
