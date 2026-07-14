#include "tensorkiln/reference.hpp"

#include <array>
#include <bit>
#include <cassert>
#include <cfenv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "float_bits.hpp"

namespace tensorkiln {
namespace {

static_assert(std::numeric_limits<float>::is_iec559);
static_assert(std::numeric_limits<double>::is_iec559);
static_assert(std::numeric_limits<double>::digits >=
              2 * std::numeric_limits<float>::digits);

template <typename... Visitors>
struct Overloaded final : Visitors... {
  using Visitors::operator()...;
};

template <typename... Visitors>
Overloaded(Visitors...) -> Overloaded<Visitors...>;

struct DenseLayout final {
  std::size_t rank = 0U;
  std::array<std::size_t, kMaxRank> extents{};
  std::array<std::size_t, kMaxRank> strides{};
  std::size_t elements = 1U;
};

struct Preflight final {
  std::vector<DenseLayout> layouts;
  std::uint64_t materialized_bytes;
  std::uint64_t scalar_steps;
};

[[nodiscard]] DenseLayout make_layout(const TensorType& type) {
  DenseLayout layout;
  layout.rank = type.shape().rank();

  std::size_t running = 1U;
  for (std::size_t remaining = layout.rank; remaining > 0U; --remaining) {
    const std::size_t axis = remaining - 1U;
    const std::int64_t raw_extent = type.shape().extents()[axis];
    assert(raw_extent > 0);
    const auto extent = static_cast<std::size_t>(raw_extent);
    assert(extent <= std::numeric_limits<std::size_t>::max() / running);
    layout.extents[axis] = extent;
    layout.strides[axis] = running;
    running *= extent;
  }

  assert(static_cast<std::uint64_t>(running) == type.numel());
  layout.elements = running;
  return layout;
}

[[nodiscard]] std::array<std::size_t, kMaxRank> unravel(
    std::size_t flat_index, const DenseLayout& layout) {
  assert(flat_index < layout.elements);
  std::array<std::size_t, kMaxRank> coordinates{};
  for (std::size_t axis = 0U; axis < layout.rank; ++axis) {
    coordinates[axis] = flat_index / layout.strides[axis];
    flat_index %= layout.strides[axis];
  }
  assert(flat_index == 0U);
  return coordinates;
}

[[nodiscard]] std::size_t broadcast_offset(
    const DenseLayout& operand, const DenseLayout& output,
    const std::array<std::size_t, kMaxRank>& output_coordinates) {
  assert(operand.rank <= output.rank);
  const std::size_t padding = output.rank - operand.rank;
  std::size_t offset = 0U;
  for (std::size_t axis = 0U; axis < operand.rank; ++axis) {
    const std::size_t coordinate =
        operand.extents[axis] == 1U
            ? 0U
            : output_coordinates[padding + axis];
    assert(coordinate < operand.extents[axis]);
    offset += coordinate * operand.strides[axis];
  }
  assert(offset < operand.elements);
  return offset;
}

[[nodiscard]] std::size_t matrix_batch_offset(
    const DenseLayout& matrix, const DenseLayout& output,
    const std::array<std::size_t, kMaxRank>& output_coordinates) {
  assert(matrix.rank >= 2U);
  assert(output.rank >= 2U);
  const std::size_t matrix_batch_rank = matrix.rank - 2U;
  const std::size_t output_batch_rank = output.rank - 2U;
  assert(matrix_batch_rank <= output_batch_rank);
  const std::size_t padding = output_batch_rank - matrix_batch_rank;

  std::size_t offset = 0U;
  for (std::size_t axis = 0U; axis < matrix_batch_rank; ++axis) {
    const std::size_t coordinate =
        matrix.extents[axis] == 1U
            ? 0U
            : output_coordinates[padding + axis];
    assert(coordinate < matrix.extents[axis]);
    offset += coordinate * matrix.strides[axis];
  }
  return offset;
}

[[nodiscard]] std::vector<float> evaluate_add(
    const Tensor& left, const DenseLayout& left_layout, const Tensor& right,
    const DenseLayout& right_layout, const DenseLayout& output_layout) {
  std::vector<float> output(output_layout.elements);
  for (std::size_t index = 0U; index < output.size(); ++index) {
    const auto coordinates = unravel(index, output_layout);
    const std::size_t left_offset =
        broadcast_offset(left_layout, output_layout, coordinates);
    const std::size_t right_offset =
        broadcast_offset(right_layout, output_layout, coordinates);
    output[index] = left.data()[left_offset] + right.data()[right_offset];
  }
  return output;
}

[[nodiscard]] std::vector<float> evaluate_matmul(
    const Tensor& left, const DenseLayout& left_layout, const Tensor& right,
    const DenseLayout& right_layout, const DenseLayout& output_layout) {
  assert(left_layout.rank >= 2U);
  assert(right_layout.rank >= 2U);
  assert(output_layout.rank >= 2U);

  const std::size_t left_row_axis = left_layout.rank - 2U;
  const std::size_t left_inner_axis = left_layout.rank - 1U;
  const std::size_t right_inner_axis = right_layout.rank - 2U;
  const std::size_t right_column_axis = right_layout.rank - 1U;
  const std::size_t output_row_axis = output_layout.rank - 2U;
  const std::size_t output_column_axis = output_layout.rank - 1U;
  const std::size_t inner = left_layout.extents[left_inner_axis];
  assert(inner == right_layout.extents[right_inner_axis]);

  std::vector<float> output(output_layout.elements);
  for (std::size_t index = 0U; index < output.size(); ++index) {
    const auto coordinates = unravel(index, output_layout);
    const std::size_t row = coordinates[output_row_axis];
    const std::size_t column = coordinates[output_column_axis];
    const std::size_t left_batch =
        matrix_batch_offset(left_layout, output_layout, coordinates);
    const std::size_t right_batch =
        matrix_batch_offset(right_layout, output_layout, coordinates);

    double accumulator = 0.0;
    for (std::size_t reduction = 0U; reduction < inner; ++reduction) {
      const std::size_t left_offset =
          left_batch + row * left_layout.strides[left_row_axis] +
          reduction * left_layout.strides[left_inner_axis];
      const std::size_t right_offset =
          right_batch + reduction * right_layout.strides[right_inner_axis] +
          column * right_layout.strides[right_column_axis];
      assert(left_offset < left_layout.elements);
      assert(right_offset < right_layout.elements);
      accumulator += static_cast<double>(left.data()[left_offset]) *
                     static_cast<double>(right.data()[right_offset]);
    }
    output[index] = static_cast<float>(accumulator);
  }
  return output;
}

[[nodiscard]] std::vector<float> evaluate_relu(const Tensor& input) {
  std::vector<float> output;
  output.reserve(input.data().size());
  for (const float value : input.data()) {
    if (std::isnan(value)) {
      output.push_back(value);
    } else {
      output.push_back(value > 0.0F ? value : 0.0F);
    }
  }
  return output;
}

[[nodiscard]] std::optional<std::size_t> find_input(
    const VerifiedGraph& graph, const std::string_view name) {
  const std::span<const Node> nodes = graph.nodes();
  for (std::size_t index = 0U; index < nodes.size(); ++index) {
    if (const auto* input = std::get_if<InputOp>(&nodes[index].operation())) {
      if (input->name == name) {
        return index;
      }
    }
  }
  return std::nullopt;
}

[[nodiscard]] Result<std::vector<const InputBinding*>> validate_bindings(
    const VerifiedGraph& graph,
    const std::span<const InputBinding> bindings) {
  std::size_t input_count = 0U;
  for (const Node& node : graph.nodes()) {
    if (std::holds_alternative<InputOp>(node.operation())) {
      ++input_count;
    }
  }
  if (bindings.size() > input_count) {
    return Result<std::vector<const InputBinding*>>::failure(Diagnostic{
        ErrorCode::input_binding_count_exceeded,
        "received " + std::to_string(bindings.size()) +
            " input bindings; graph defines " + std::to_string(input_count),
    });
  }

  for (std::size_t index = 0U; index < bindings.size(); ++index) {
    const InputBinding& binding = bindings[index];
    const auto node_index = find_input(graph, binding.name);
    if (!node_index.has_value()) {
      return Result<std::vector<const InputBinding*>>::failure(Diagnostic{
          ErrorCode::input_binding_unknown,
          "input binding #" + std::to_string(index) +
              " does not match any graph input",
      });
    }
  }

  std::vector<const InputBinding*> selected(graph.nodes().size(), nullptr);
  for (std::size_t index = 0U; index < bindings.size(); ++index) {
    const auto found = find_input(graph, bindings[index].name);
    assert(found.has_value());
    const std::size_t node_index = *found;
    if (selected[node_index] != nullptr) {
      return Result<std::vector<const InputBinding*>>::failure(Diagnostic{
          ErrorCode::input_binding_duplicate,
          "graph input is bound more than once: " +
              std::string(bindings[index].name),
      });
    }
    selected[node_index] = &bindings[index];
  }

  const std::span<const Node> nodes = graph.nodes();
  for (std::size_t index = 0U; index < nodes.size(); ++index) {
    const auto* input = std::get_if<InputOp>(&nodes[index].operation());
    if (input != nullptr && selected[index] == nullptr) {
      return Result<std::vector<const InputBinding*>>::failure(Diagnostic{
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
    assert(selected[index] != nullptr);
    const std::size_t expected =
        nodes[index].output_type().byte_count() / sizeof(float);
    const std::size_t received = selected[index]->data.size();
    if (received != expected) {
      return Result<std::vector<const InputBinding*>>::failure(Diagnostic{
          ErrorCode::input_binding_size_mismatch,
          "input " + input->name + " expects " + std::to_string(expected) +
              " f32 values; received " + std::to_string(received),
      });
    }
  }

  return Result<std::vector<const InputBinding*>>::success(
      std::move(selected));
}

[[nodiscard]] bool add_within_limit(std::uint64_t& total,
                                    const std::uint64_t amount,
                                    const std::uint64_t limit) noexcept {
  if (total > limit || amount > limit - total) {
    return false;
  }
  total += amount;
  return true;
}

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

[[nodiscard]] Result<Preflight> preflight(const VerifiedGraph& graph,
                                          const ReferenceLimits limits) {
  Preflight result{{}, 0U, 0U};
  result.layouts.reserve(graph.nodes().size());
  for (const Node& node : graph.nodes()) {
    result.layouts.push_back(make_layout(node.output_type()));
  }

  for (const Node& node : graph.nodes()) {
    const auto bytes =
        static_cast<std::uint64_t>(node.output_type().byte_count());
    if (!add_within_limit(result.materialized_bytes, bytes,
                          limits.max_materialized_bytes)) {
      return Result<Preflight>::failure(Diagnostic{
          ErrorCode::reference_materialization_limit_exceeded,
          "materializing node #" + std::to_string(node.id().ordinal()) +
              " would exceed reference byte limit " +
              std::to_string(limits.max_materialized_bytes),
      });
    }
  }

  for (const Node& node : graph.nodes()) {
    const std::uint64_t elements = node.output_type().numel();
    const std::optional<std::uint64_t> work = std::visit(
        Overloaded{
            [elements](const InputOp&) {
              return std::optional<std::uint64_t>{elements};
            },
            [elements](const ConstantOp&) {
              return std::optional<std::uint64_t>{elements};
            },
            [elements](const AddOp&) {
              return std::optional<std::uint64_t>{elements};
            },
            [&, elements](const MatMulOp&) {
              assert(node.inputs().size() == 2U);
              const TensorType* left = graph.type(node.inputs()[0]);
              assert(left != nullptr);
              const auto inner = static_cast<std::uint64_t>(
                  left->shape().extents()[left->shape().rank() - 1U]);
              if (inner != 0U &&
                  elements >
                      std::numeric_limits<std::uint64_t>::max() / inner) {
                return std::optional<std::uint64_t>{};
              }
              return std::optional<std::uint64_t>{elements * inner};
            },
            [elements](const ReluOp&) {
              return std::optional<std::uint64_t>{elements};
            },
        },
        node.operation());

    if (!work.has_value() ||
        !add_within_limit(result.scalar_steps, work.value_or(0U),
                          limits.max_scalar_steps)) {
      return Result<Preflight>::failure(Diagnostic{
          ErrorCode::reference_scalar_step_limit_exceeded,
          "evaluating node #" + std::to_string(node.id().ordinal()) +
              " would exceed reference scalar-step limit " +
              std::to_string(limits.max_scalar_steps),
      });
    }
  }

  return Result<Preflight>::success(std::move(result));
}

}  // namespace

Tensor::Tensor(TensorType type, std::vector<float> data)
    : type_(std::move(type)), data_(std::move(data)) {}

ReferenceResult::ReferenceResult(std::vector<ValueId> value_ids,
                                 std::vector<Tensor> values,
                                 std::vector<OutputRecord> outputs,
                                 const std::uint64_t materialized_bytes,
                                 const std::uint64_t scalar_steps)
    : value_ids_(std::move(value_ids)),
      values_(std::move(values)),
      outputs_(std::move(outputs)),
      materialized_bytes_(materialized_bytes),
      scalar_steps_(scalar_steps) {
  assert(value_ids_.size() == values_.size());
}

const Tensor* ReferenceResult::value(const ValueId id) const noexcept {
  const std::size_t index = static_cast<std::size_t>(id.ordinal());
  if (index >= value_ids_.size() || value_ids_[index] != id) {
    return nullptr;
  }
  return &values_[index];
}

const Tensor* ReferenceResult::output(const std::string_view name) const
    noexcept {
  for (const OutputRecord& output_record : outputs_) {
    if (output_record.name == name) {
      return value(output_record.value);
    }
  }
  return nullptr;
}

Result<ReferenceResult> ReferenceInterpreter::run(
    const VerifiedGraph& graph, const std::span<const InputBinding> bindings,
    const ReferenceLimits limits) {
  auto validated_bindings = validate_bindings(graph, bindings);
  if (!validated_bindings.has_value()) {
    return Result<ReferenceResult>::failure(*validated_bindings.error_if());
  }

  if (std::fegetround() != FE_TONEAREST) {
    return Result<ReferenceResult>::failure(Diagnostic{
        ErrorCode::unsupported_rounding_mode,
        "reference interpreter requires the FE_TONEAREST rounding mode",
    });
  }
  if (!gradual_underflow_is_active()) {
    return Result<ReferenceResult>::failure(Diagnostic{
        ErrorCode::unsupported_subnormal_mode,
        "reference interpreter requires gradual f32 underflow; FTZ or DAZ "
        "appears active",
    });
  }

  auto checked = preflight(graph, limits);
  if (!checked.has_value()) {
    return Result<ReferenceResult>::failure(*checked.error_if());
  }

  const std::vector<const InputBinding*>& selected =
      *validated_bindings.value_if();
  Preflight execution = std::move(*checked.value_if());
  std::vector<Tensor> values;
  values.reserve(graph.nodes().size());

  const std::span<const Node> nodes = graph.nodes();
  for (std::size_t node_index = 0U; node_index < nodes.size(); ++node_index) {
    const Node& node = nodes[node_index];
    std::vector<float> data = std::visit(
        Overloaded{
            [&](const InputOp&) {
              assert(selected[node_index] != nullptr);
              return detail::copy_float_bits(selected[node_index]->data);
            },
            [](const ConstantOp& constant) {
              return detail::copy_float_bits(
                  std::span<const float>{constant.data});
            },
            [&](const AddOp&) {
              assert(node.inputs().size() == 2U);
              const std::size_t left_index =
                  static_cast<std::size_t>(node.inputs()[0].ordinal());
              const std::size_t right_index =
                  static_cast<std::size_t>(node.inputs()[1].ordinal());
              assert(left_index < values.size());
              assert(right_index < values.size());
              return evaluate_add(
                  values[left_index], execution.layouts[left_index],
                  values[right_index], execution.layouts[right_index],
                  execution.layouts[node_index]);
            },
            [&](const MatMulOp&) {
              assert(node.inputs().size() == 2U);
              const std::size_t left_index =
                  static_cast<std::size_t>(node.inputs()[0].ordinal());
              const std::size_t right_index =
                  static_cast<std::size_t>(node.inputs()[1].ordinal());
              assert(left_index < values.size());
              assert(right_index < values.size());
              return evaluate_matmul(
                  values[left_index], execution.layouts[left_index],
                  values[right_index], execution.layouts[right_index],
                  execution.layouts[node_index]);
            },
            [&](const ReluOp&) {
              assert(node.inputs().size() == 1U);
              const std::size_t input_index =
                  static_cast<std::size_t>(node.inputs()[0].ordinal());
              assert(input_index < values.size());
              return evaluate_relu(values[input_index]);
            },
        },
        node.operation());
    assert(data.size() == execution.layouts[node_index].elements);
    Tensor value(node.output_type(), std::move(data));
    values.push_back(std::move(value));
  }

  std::vector<ValueId> value_ids;
  value_ids.reserve(nodes.size());
  for (const Node& node : nodes) {
    value_ids.push_back(node.output());
  }

  std::vector<ReferenceResult::OutputRecord> outputs;
  outputs.reserve(graph.outputs().size());
  for (const GraphOutput& output : graph.outputs()) {
    outputs.push_back(ReferenceResult::OutputRecord{output.name(),
                                                    output.value()});
  }

  return Result<ReferenceResult>::success(ReferenceResult(
      std::move(value_ids), std::move(values), std::move(outputs),
      execution.materialized_bytes, execution.scalar_steps));
}

}  // namespace tensorkiln
