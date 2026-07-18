#include "tensorkiln/execution.hpp"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cfenv>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "aligned_workspace.hpp"
#include "execution_internal.hpp"
#include "execution_kernels.hpp"

namespace tensorkiln {
namespace {

[[nodiscard]] bool gradual_underflow_is_active() noexcept {
  const volatile float denormal_input =
      std::numeric_limits<float>::denorm_min();
  const volatile float zero = 0.0F;
  const volatile float minimum_normal = std::numeric_limits<float>::min();
  const volatile float half = 0.5F;

  const float consumed = denormal_input + zero;
  const float produced = minimum_normal * half;
  return std::bit_cast<std::uint32_t>(consumed) == 0x00000001U &&
         std::bit_cast<std::uint32_t>(produced) == 0x00400000U;
}

[[nodiscard]] bool binary64_precision_is_active() noexcept {
  const volatile double base = 1073741824.0;
  const volatile double one = 1.0;
  const volatile double sum = base + one;
  return sum == 1073741825.0;
}

[[nodiscard]] bool binary32_rounding_is_nearest() noexcept {
  const volatile float one = 1.0F;
  const volatile float tie = 0x1p-24F;
  const volatile float above_tie = 0x1.8p-24F;
  const volatile float positive_tie = one + tie;
  const volatile float negative_tie = -one - tie;
  const volatile float positive_above_tie = one + above_tie;
  return positive_tie == 1.0F && negative_tie == -1.0F &&
         std::bit_cast<std::uint32_t>(
             static_cast<float>(positive_above_tie)) == UINT32_C(0x3f800001);
}

[[nodiscard]] std::optional<std::size_t> find_input_index(
    const ExecutionPlan& plan, const std::string_view name) noexcept {
  const std::span<const Node> nodes = plan.graph().nodes();
  for (std::size_t index = 0U; index < nodes.size(); ++index) {
    const auto* input = std::get_if<InputOp>(&nodes[index].operation());
    if (input != nullptr && input->name == name) {
      return index;
    }
  }
  return std::nullopt;
}

[[nodiscard]] std::uint64_t next_generation(
    const std::uint64_t generation) noexcept {
  return generation == std::numeric_limits<std::uint64_t>::max()
             ? 1U
             : generation + 1U;
}

}  // namespace

namespace detail {

struct ExecutionResultState final {
  const ExecutionSession* session = nullptr;
};

class ExecutionSessionData final {
 public:
  ExecutionSessionData(const ExecutionPlan& execution_plan,
                       const ExecutionSessionOptions options)
      : plan(&execution_plan),
        workspace(execution_plan.stats().workspace_bytes),
        value_data(execution_plan.values().size(), nullptr),
        selected_bindings(execution_plan.values().size(), nullptr),
        result_state(std::make_shared<ExecutionResultState>()),
        audit_kernel_writes(options.audit_kernel_writes),
        shadow_workspace(options.audit_kernel_writes
                             ? workspace.bytes().size()
                             : 0U) {
    assert(execution_plan.values().size() ==
           execution_plan.graph().nodes().size());
    const std::span<const std::byte> workspace_bytes = workspace.bytes();
    for (const PlanValue& value : execution_plan.values()) {
      const std::size_t value_index =
          static_cast<std::size_t>(value.source_value().ordinal());
      assert(value_index < value_data.size());
      assert(execution_plan.values()[value_index].source_value() ==
             value.source_value());
      const Node& source_node = execution_plan.graph().nodes()[value_index];

      switch (value.storage().kind()) {
        case PlanStorageKind::input:
          ++input_count;
          break;
        case PlanStorageKind::constant: {
          const auto* constant =
              std::get_if<ConstantOp>(&source_node.operation());
          assert(constant != nullptr);
          assert(constant->data.size() == value.layout().elements());
          value_data[value_index] = constant->data.data();
          break;
        }
        case PlanStorageKind::arena: {
          const std::uint64_t offset = value.storage().offset_bytes();
          [[maybe_unused]] const std::uint64_t payload =
              static_cast<std::uint64_t>(value.type().byte_count());
          assert(offset <= workspace.logical_bytes());
          assert(payload <= workspace.logical_bytes() - offset);
          assert(offset % kArenaAlignmentBytes == 0U);
          const std::size_t byte_offset = static_cast<std::size_t>(offset);
          assert(byte_offset <= workspace_bytes.size());
          value_data[value_index] = reinterpret_cast<const float*>(
              workspace_bytes.data() + byte_offset);
          break;
        }
      }
    }
    assert(input_count == execution_plan.stats().input_count);
  }

  [[nodiscard]] float* writable_output(const ValueId output) noexcept {
    const std::size_t index = static_cast<std::size_t>(output.ordinal());
    assert(index < plan->values().size());
    const PlanValue& value = plan->values()[index];
    assert(value.source_value() == output);
    assert(value.storage().kind() == PlanStorageKind::arena);
    const std::uint64_t offset = value.storage().offset_bytes();
    assert(offset <= workspace.logical_bytes());
    return reinterpret_cast<float*>(
        workspace.bytes().data() + static_cast<std::size_t>(offset));
  }

  [[nodiscard]] bool write_set_is_intact(
      const PlanValue& output) const noexcept {
    if (!audit_kernel_writes) {
      return true;
    }
    const std::span<const std::byte> actual = workspace.bytes();
    assert(shadow_workspace.size() == actual.size());
    const std::size_t begin =
        static_cast<std::size_t>(output.storage().offset_bytes());
    const std::size_t payload = output.type().byte_count();
    assert(begin <= actual.size());
    assert(payload <= actual.size() - begin);
    const std::size_t end = begin + payload;
    for (std::size_t index = 0U; index < actual.size(); ++index) {
      if ((index < begin || index >= end) &&
          actual[index] != shadow_workspace[index]) {
        return false;
      }
    }
    return true;
  }

  void inject_fault(const PlanValue& output) noexcept {
    if (fault != ExecutionFaultKind::write_outside_output) {
      return;
    }
    fault = ExecutionFaultKind::none;
    assert(audit_kernel_writes);
    if (!audit_kernel_writes) {
      return;
    }
    std::span<std::byte> actual = workspace.bytes();
    const std::size_t begin =
        static_cast<std::size_t>(output.storage().offset_bytes());
    const std::size_t end = begin + output.type().byte_count();
    if (end < actual.size()) {
      actual[end] ^= std::byte{0x01U};
    } else if (begin > 0U) {
      actual[begin - 1U] ^= std::byte{0x01U};
    } else {
      workspace.corrupt_suffix_for_test();
    }
  }

  const ExecutionPlan* plan;
  AlignedWorkspace workspace;
  std::vector<const float*> value_data;
  std::vector<const ExecutionInputBinding*> selected_bindings;
  std::shared_ptr<ExecutionResultState> result_state;
  bool audit_kernel_writes;
  std::vector<std::byte> shadow_workspace;
  ExecutionFaultKind fault = ExecutionFaultKind::none;
  std::uint32_t input_count = 0U;
  std::uint64_t generation = 0U;
  bool inputs_bound = false;
  bool result_valid = false;
};

}  // namespace detail

std::string_view execution_run_status_name(
    const ExecutionRunStatus status) noexcept {
  switch (status) {
    case ExecutionRunStatus::success:
      return "success";
    case ExecutionRunStatus::inputs_not_bound:
      return "inputs_not_bound";
    case ExecutionRunStatus::unsupported_rounding_mode:
      return "unsupported_rounding_mode";
    case ExecutionRunStatus::unsupported_binary64_precision:
      return "unsupported_binary64_precision";
    case ExecutionRunStatus::unsupported_subnormal_mode:
      return "unsupported_subnormal_mode";
    case ExecutionRunStatus::memory_corruption:
      return "memory_corruption";
  }
  return "unknown_execution_run_status";
}

bool ExecutionResultView::current() const noexcept {
  const std::shared_ptr<const detail::ExecutionResultState> state =
      state_.lock();
  return state != nullptr && state->session != nullptr &&
         state->session->result_is_current(generation_);
}

std::optional<TensorView> ExecutionResultView::output(
    const std::string_view name) const noexcept {
  const std::shared_ptr<const detail::ExecutionResultState> state =
      state_.lock();
  if (state == nullptr || state->session == nullptr) {
    return std::nullopt;
  }
  return state->session->output(name, generation_);
}

ExecutionSession ExecutionSession::create(
    const ExecutionPlan& plan, const ExecutionSessionOptions options) {
  return ExecutionSession(
      std::make_unique<detail::ExecutionSessionData>(plan, options));
}

ExecutionSession::ExecutionSession(
    std::unique_ptr<detail::ExecutionSessionData> data) noexcept
    : data_(std::move(data)) {
  assert(data_ != nullptr);
  assert(data_->result_state != nullptr);
  data_->result_state->session = this;
}

ExecutionSession::ExecutionSession(ExecutionSession&& other) noexcept
    : data_(std::move(other.data_)) {
  if (data_ != nullptr) {
    data_->generation = next_generation(data_->generation);
    data_->result_valid = false;
    assert(data_->result_state != nullptr);
    data_->result_state->session = this;
  }
}

ExecutionSession::~ExecutionSession() = default;

const ExecutionPlan& ExecutionSession::plan() const noexcept {
  assert(data_ != nullptr);
  assert(data_->plan != nullptr);
  return *data_->plan;
}

std::uint64_t ExecutionSession::workspace_bytes() const noexcept {
  assert(data_ != nullptr);
  return data_->workspace.logical_bytes();
}

bool ExecutionSession::audits_kernel_writes() const noexcept {
  assert(data_ != nullptr);
  return data_->audit_kernel_writes;
}

Result<BoundInputs> ExecutionSession::bind(
    const std::span<const ExecutionInputBinding> bindings) {
  assert(data_ != nullptr);
  data_->generation = next_generation(data_->generation);
  data_->inputs_bound = false;
  data_->result_valid = false;
  std::fill(data_->selected_bindings.begin(),
            data_->selected_bindings.end(), nullptr);

  if (bindings.size() > static_cast<std::size_t>(data_->input_count)) {
    return Result<BoundInputs>::failure(Diagnostic{
        ErrorCode::input_binding_count_exceeded,
        "received " + std::to_string(bindings.size()) +
            " input bindings; graph defines " +
            std::to_string(data_->input_count),
    });
  }

  for (std::size_t index = 0U; index < bindings.size(); ++index) {
    if (!find_input_index(*data_->plan, bindings[index].name).has_value()) {
      return Result<BoundInputs>::failure(Diagnostic{
          ErrorCode::input_binding_unknown,
          "input binding #" + std::to_string(index) +
              " does not match any graph input",
      });
    }
  }

  for (const ExecutionInputBinding& binding : bindings) {
    const std::optional<std::size_t> found =
        find_input_index(*data_->plan, binding.name);
    assert(found.has_value());
    if (data_->selected_bindings[*found] != nullptr) {
      return Result<BoundInputs>::failure(Diagnostic{
          ErrorCode::input_binding_duplicate,
          "graph input is bound more than once: " +
              std::string(binding.name),
      });
    }
    data_->selected_bindings[*found] = &binding;
  }

  const std::span<const Node> nodes = data_->plan->graph().nodes();
  for (std::size_t index = 0U; index < nodes.size(); ++index) {
    const auto* input = std::get_if<InputOp>(&nodes[index].operation());
    if (input != nullptr && data_->selected_bindings[index] == nullptr) {
      return Result<BoundInputs>::failure(Diagnostic{
          ErrorCode::input_binding_missing,
          "graph input has no binding: " + input->name,
      });
    }
  }

  for (std::size_t index = 0U; index < nodes.size(); ++index) {
    const auto* input = std::get_if<InputOp>(&nodes[index].operation());
    if (input == nullptr) {
      continue;
    }
    assert(data_->selected_bindings[index] != nullptr);
    const std::size_t expected =
        nodes[index].output_type().byte_count() / sizeof(float);
    const std::size_t received =
        data_->selected_bindings[index]->data.size();
    if (received != expected) {
      return Result<BoundInputs>::failure(Diagnostic{
          ErrorCode::input_binding_size_mismatch,
          "input " + input->name + " expects " +
              std::to_string(expected) + " f32 values; received " +
              std::to_string(received),
      });
    }
  }

  const std::span<const std::byte> workspace = data_->workspace.bytes();
  if (!workspace.empty()) {
    const std::uintptr_t workspace_begin =
        reinterpret_cast<std::uintptr_t>(workspace.data());
    if (workspace.size() >
        std::numeric_limits<std::uintptr_t>::max() - workspace_begin) {
      return Result<BoundInputs>::failure(Diagnostic{
          ErrorCode::compiler_internal_invariant,
          "execution workspace has an unrepresentable address range",
      });
    }
    const std::uintptr_t workspace_end = workspace_begin + workspace.size();
    for (std::size_t index = 0U; index < nodes.size(); ++index) {
      if (!std::holds_alternative<InputOp>(nodes[index].operation())) {
        continue;
      }
      assert(data_->selected_bindings[index] != nullptr);
      const std::span<const float> payload =
          data_->selected_bindings[index]->data;
      const std::uintptr_t payload_begin =
          reinterpret_cast<std::uintptr_t>(payload.data());
      const std::size_t payload_bytes = payload.size() * sizeof(float);
      if (payload_begin >
          std::numeric_limits<std::uintptr_t>::max() - payload_bytes) {
        return Result<BoundInputs>::failure(Diagnostic{
            ErrorCode::input_binding_aliases_workspace,
            "input " + std::get<InputOp>(nodes[index].operation()).name +
                " has an unrepresentable address range",
        });
      }
      const std::uintptr_t payload_end = payload_begin + payload_bytes;
      if (payload_begin < workspace_end && workspace_begin < payload_end) {
        return Result<BoundInputs>::failure(Diagnostic{
            ErrorCode::input_binding_aliases_workspace,
            "input " + std::get<InputOp>(nodes[index].operation()).name +
                " aliases this execution session's workspace",
        });
      }
    }
  }

  for (std::size_t index = 0U; index < nodes.size(); ++index) {
    if (std::holds_alternative<InputOp>(nodes[index].operation())) {
      assert(data_->selected_bindings[index] != nullptr);
      data_->value_data[index] =
          data_->selected_bindings[index]->data.data();
    }
  }
  data_->inputs_bound = true;
  return Result<BoundInputs>::success(BoundInputs(data_->input_count));
}

ExecutionRunStatus ExecutionSession::run() noexcept {
  assert(data_ != nullptr);
  data_->generation = next_generation(data_->generation);
  data_->result_valid = false;
  if (!data_->inputs_bound) {
    return ExecutionRunStatus::inputs_not_bound;
  }
  if (!data_->workspace.guards_intact()) {
    return ExecutionRunStatus::memory_corruption;
  }
  if (std::fegetround() != FE_TONEAREST ||
      !binary32_rounding_is_nearest()) {
    return ExecutionRunStatus::unsupported_rounding_mode;
  }
  if (!binary64_precision_is_active()) {
    return ExecutionRunStatus::unsupported_binary64_precision;
  }
  if (!gradual_underflow_is_active()) {
    return ExecutionRunStatus::unsupported_subnormal_mode;
  }

  for (const ExecutionStep& step : data_->plan->steps()) {
    if (data_->audit_kernel_writes) {
      const std::span<const std::byte> workspace = data_->workspace.bytes();
      assert(data_->shadow_workspace.size() == workspace.size());
      std::copy(workspace.begin(), workspace.end(),
                data_->shadow_workspace.begin());
    }
    float* const output = data_->writable_output(step.output());
    detail::execute_dense_kernel(step, data_->plan->values(),
                                 data_->value_data, output);
    const PlanValue* const output_value = data_->plan->value(step.output());
    assert(output_value != nullptr);
    data_->inject_fault(*output_value);
    if (!data_->write_set_is_intact(*output_value)) {
      return ExecutionRunStatus::memory_corruption;
    }
    if (!data_->workspace.guards_intact()) {
      return ExecutionRunStatus::memory_corruption;
    }
  }
  data_->result_valid = true;
  return ExecutionRunStatus::success;
}

std::optional<ExecutionResultView> ExecutionSession::result() const noexcept {
  if (data_ == nullptr || !data_->result_valid) {
    return std::nullopt;
  }
  return ExecutionResultView(data_->result_state, data_->generation);
}

bool ExecutionSession::result_is_current(
    const std::uint64_t generation) const noexcept {
  return data_ != nullptr && data_->result_valid &&
         data_->generation == generation;
}

std::optional<TensorView> ExecutionSession::output(
    const std::string_view name, const std::uint64_t generation) const
    noexcept {
  if (!result_is_current(generation)) {
    return std::nullopt;
  }
  assert(data_ != nullptr);
  for (const GraphOutput& output_record : data_->plan->graph().outputs()) {
    if (output_record.name() != name) {
      continue;
    }
    const PlanValue* const value =
        data_->plan->value(output_record.value());
    if (value == nullptr) {
      return std::nullopt;
    }
    const std::size_t index =
        static_cast<std::size_t>(value->source_value().ordinal());
    assert(index < data_->value_data.size());
    const float* const payload = data_->value_data[index];
    assert(payload != nullptr);
    return TensorView(
        value->type(),
        std::span<const float>{
            payload, static_cast<std::size_t>(value->layout().elements())});
  }
  return std::nullopt;
}

namespace detail {

void set_execution_fault_for_test(
    ExecutionSession& session, const ExecutionFaultKind fault) noexcept {
  assert(session.data_ != nullptr);
  assert(fault == ExecutionFaultKind::none ||
         session.data_->audit_kernel_writes);
  session.data_->fault = fault;
}

}  // namespace detail

}  // namespace tensorkiln
