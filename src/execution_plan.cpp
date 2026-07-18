#include "tensorkiln/execution_plan.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <variant>

namespace tensorkiln {

std::string_view plan_storage_kind_name(const PlanStorageKind kind) noexcept {
  switch (kind) {
    case PlanStorageKind::input:
      return "input";
    case PlanStorageKind::constant:
      return "constant";
    case PlanStorageKind::arena:
      return "arena";
  }
  return "invalid";
}

std::string_view dense_kernel_kind_name(const DenseKernelKind kind) noexcept {
  switch (kind) {
    case DenseKernelKind::add_contiguous_f32:
      return "add_contiguous_f32";
    case DenseKernelKind::add_broadcast_f32:
      return "add_broadcast_f32";
    case DenseKernelKind::matmul_rank2_f32:
      return "matmul_rank2_f32";
    case DenseKernelKind::matmul_batched_f32:
      return "matmul_batched_f32";
    case DenseKernelKind::relu_contiguous_f32:
      return "relu_contiguous_f32";
  }
  return "invalid";
}

DenseLayout::DenseLayout(
    std::array<std::uint64_t, kMaxRank> strides_elements,
    const std::size_t rank, const std::uint64_t elements) noexcept
    : strides_elements_(strides_elements), rank_(rank), elements_(elements) {}

PlanStorage::PlanStorage(const PlanStorageKind kind,
                         const std::uint32_t ordinal,
                         const std::uint64_t offset_bytes) noexcept
    : kind_(kind), ordinal_(ordinal), offset_bytes_(offset_bytes) {}

PlanValue::PlanValue(const ValueId source_value, TensorType type,
                     DenseLayout layout, const PlanStorage storage)
    : source_value_(source_value),
      type_(std::move(type)),
      layout_(layout),
      storage_(storage) {}

ExecutionStep::ExecutionStep(const std::uint32_t ordinal,
                             const NodeId source_node,
                             const DenseKernelKind kernel,
                             std::vector<ValueId> operands,
                             const ValueId output,
                             const std::uint64_t scalar_steps)
    : ordinal_(ordinal),
      source_node_(source_node),
      kernel_(kernel),
      operands_(std::move(operands)),
      output_(output),
      scalar_steps_(scalar_steps) {}

ExecutionPlan::ExecutionPlan(ExecutionPlanLimits limits,
                             const ExecutionPlanStats stats,
                             VerifiedGraph graph,
                             GraphArenaLoweringResult arena,
                             std::vector<PlanValue> values,
                             std::vector<ExecutionStep> steps)
    : limits_(limits),
      stats_(stats),
      graph_(std::move(graph)),
      arena_(std::move(arena)),
      values_(std::move(values)),
      steps_(std::move(steps)) {}

const PlanValue* ExecutionPlan::value(const ValueId source_value) const
    noexcept {
  if (graph_.type(source_value) == nullptr) {
    return nullptr;
  }
  const std::size_t index = static_cast<std::size_t>(source_value.ordinal());
  if (index >= values_.size() ||
      values_[index].source_value() != source_value) {
    return nullptr;
  }
  return &values_[index];
}

std::string ExecutionPlan::dump() const {
  std::string result{"tensorkiln.execution_plan v0 {\n"};
  result += "  limits {values=" + std::to_string(limits_.max_values) +
            ", steps=" + std::to_string(limits_.max_steps) +
            ", outputs=" + std::to_string(limits_.max_outputs) +
            ", constant_bytes=" +
            std::to_string(limits_.max_owned_constant_bytes) +
            ", scalar_steps=" + std::to_string(limits_.max_scalar_steps) +
            ", arena_buffers=" +
            std::to_string(limits_.arena_limits.max_buffers) +
            ", arena_workspace_bytes=" +
            std::to_string(limits_.arena_limits.max_workspace_bytes) +
            "}\n";
  result += "  stats {values=" + std::to_string(stats_.value_count) +
            ", inputs=" + std::to_string(stats_.input_count) +
            ", constants=" + std::to_string(stats_.constant_count) +
            ", steps=" + std::to_string(stats_.step_count) +
            ", outputs=" + std::to_string(stats_.output_count) +
            ", constant_bytes=" +
            std::to_string(stats_.owned_constant_bytes) +
            ", scalar_steps=" + std::to_string(stats_.scalar_steps) +
            ", workspace_bytes=" +
            std::to_string(stats_.workspace_bytes) + "}\n";
  result += "  values {\n";
  const std::span<const Node> nodes = graph_.nodes();
  for (std::size_t index = 0U; index < values_.size(); ++index) {
    const PlanValue& value_record = values_[index];
    const Node& node = nodes[index];
    result += "    %" + std::to_string(value_record.source_value().ordinal()) +
              " " + value_record.type().to_string() + " dense strides=[";
    const std::span<const std::uint64_t> strides =
        value_record.layout().strides_elements();
    for (std::size_t axis = 0U; axis < strides.size(); ++axis) {
      if (axis != 0U) {
        result += ",";
      }
      result += std::to_string(strides[axis]);
    }
    result += "] storage=" +
              std::string(plan_storage_kind_name(
                  value_record.storage().kind()));
    switch (value_record.storage().kind()) {
      case PlanStorageKind::input:
        result += " #i" +
                  std::to_string(value_record.storage().ordinal());
        break;
      case PlanStorageKind::constant:
        result += " #c" +
                  std::to_string(value_record.storage().ordinal());
        break;
      case PlanStorageKind::arena:
        result += " #b" +
                  std::to_string(value_record.storage().ordinal()) +
                  " offset=" +
                  std::to_string(value_record.storage().offset_bytes());
        break;
    }
    if (const auto* input = std::get_if<InputOp>(&node.operation())) {
      result += " name=" + input->name;
    } else if (const auto* constant =
                   std::get_if<ConstantOp>(&node.operation())) {
      result += " name=" + constant->name + " fingerprint=" +
                std::to_string(constant->fingerprint);
    }
    result += "\n";
  }
  result += "  }\n  steps {\n";
  for (const ExecutionStep& step : steps_) {
    result += "    @" + std::to_string(step.ordinal()) + " #n" +
              std::to_string(step.source_node().ordinal()) + " %" +
              std::to_string(step.output().ordinal()) + " = " +
              std::string(dense_kernel_kind_name(step.kernel())) + "(";
    for (std::size_t index = 0U; index < step.operands().size(); ++index) {
      if (index != 0U) {
        result += ",";
      }
      result += "%" + std::to_string(step.operands()[index].ordinal());
    }
    result += ") work=" + std::to_string(step.scalar_steps()) + "\n";
  }
  result += "  }\n  outputs {\n";
  for (const GraphOutput& output : graph_.outputs()) {
    result += "    #o" + std::to_string(output.id().ordinal) + " " +
              output.name() + " -> %" +
              std::to_string(output.value().ordinal()) + "\n";
  }
  result += "  }\n  arena {\n";
  for (const ArenaAllocation& allocation : arena_.arena_plan().allocations()) {
    result += "    #b" + std::to_string(allocation.buffer_ordinal()) +
              " offset=" + std::to_string(allocation.offset_bytes()) +
              " payload=" + std::to_string(allocation.payload_bytes()) +
              " reserved=" + std::to_string(allocation.reserved_bytes()) +
              " live=[" + std::to_string(allocation.live_begin_step()) +
              "," +
              std::to_string(allocation.live_end_step_exclusive()) +
              ")\n";
  }
  result += "  }\n}\n";
  return result;
}

}  // namespace tensorkiln
