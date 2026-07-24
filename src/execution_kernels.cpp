#include "execution_kernels.hpp"

#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>

namespace tensorkiln::detail {
namespace {

// i386 x87 may otherwise retain a nominal double accumulator in extended
// precision. Other targets keep the reduction in registers.
[[nodiscard]] double round_to_binary64(const double value) noexcept {
#if defined(__FLT_EVAL_METHOD__) && __FLT_EVAL_METHOD__ == 2
  const volatile double rounded = value;
  return rounded;
#else
  return value;
#endif
}

[[nodiscard]] const PlanValue& value_at(
    const std::span<const PlanValue> values, const ValueId value) noexcept {
  const std::size_t index = static_cast<std::size_t>(value.ordinal());
  assert(index < values.size());
  assert(values[index].source_value() == value);
  return values[index];
}

[[nodiscard]] const float* data_at(
    const std::span<const float* const> value_data,
    const ValueId value) noexcept {
  const std::size_t index = static_cast<std::size_t>(value.ordinal());
  assert(index < value_data.size());
  assert(value_data[index] != nullptr);
  return value_data[index];
}

[[nodiscard]] std::uint64_t extent(const PlanValue& value,
                                   const std::size_t axis) noexcept {
  assert(axis < value.type().shape().rank());
  const std::int64_t raw = value.type().shape().extents()[axis];
  assert(raw > 0);
  return static_cast<std::uint64_t>(raw);
}

[[nodiscard]] std::array<std::uint64_t, kMaxRank> unravel(
    std::uint64_t flat_index, const PlanValue& value) noexcept {
  assert(flat_index < value.layout().elements());
  const std::span<const std::uint64_t> strides =
      value.layout().strides_elements();
  std::array<std::uint64_t, kMaxRank> coordinates{};
  for (std::size_t axis = 0U; axis < value.layout().rank(); ++axis) {
    assert(strides[axis] > 0U);
    coordinates[axis] = flat_index / strides[axis];
    flat_index %= strides[axis];
  }
  assert(flat_index == 0U);
  return coordinates;
}

[[nodiscard]] std::uint64_t broadcast_offset(
    const PlanValue& operand, const PlanValue& output,
    const std::array<std::uint64_t, kMaxRank>& coordinates) noexcept {
  assert(operand.layout().rank() <= output.layout().rank());
  const std::size_t padding =
      output.layout().rank() - operand.layout().rank();
  const std::span<const std::uint64_t> strides =
      operand.layout().strides_elements();
  std::uint64_t offset = 0U;
  for (std::size_t axis = 0U; axis < operand.layout().rank(); ++axis) {
    const std::uint64_t coordinate =
        extent(operand, axis) == 1U ? 0U : coordinates[padding + axis];
    assert(coordinate < extent(operand, axis));
    offset += coordinate * strides[axis];
  }
  assert(offset < operand.layout().elements());
  return offset;
}

[[nodiscard]] std::uint64_t matrix_batch_offset(
    const PlanValue& matrix, const PlanValue& output,
    const std::array<std::uint64_t, kMaxRank>& coordinates) noexcept {
  assert(matrix.layout().rank() >= 2U);
  assert(output.layout().rank() >= 2U);
  const std::size_t matrix_batch_rank = matrix.layout().rank() - 2U;
  const std::size_t output_batch_rank = output.layout().rank() - 2U;
  assert(matrix_batch_rank <= output_batch_rank);
  const std::size_t padding = output_batch_rank - matrix_batch_rank;
  const std::span<const std::uint64_t> strides =
      matrix.layout().strides_elements();

  std::uint64_t offset = 0U;
  for (std::size_t axis = 0U; axis < matrix_batch_rank; ++axis) {
    const std::uint64_t coordinate =
        extent(matrix, axis) == 1U ? 0U : coordinates[padding + axis];
    assert(coordinate < extent(matrix, axis));
    offset += coordinate * strides[axis];
  }
  return offset;
}

void execute_add_contiguous(
    const ExecutionStep& step, const std::span<const PlanValue> values,
    const std::span<const float* const> value_data,
    float* const output_data) noexcept {
  assert(step.operands().size() == 2U);
  [[maybe_unused]] const PlanValue& output =
      value_at(values, step.output());
  const PlanValue& left = value_at(values, step.operands()[0]);
  const PlanValue& right = value_at(values, step.operands()[1]);
  assert(left.type() == output.type());
  assert(right.type() == output.type());
  const float* const left_data = data_at(value_data, left.source_value());
  const float* const right_data = data_at(value_data, right.source_value());

  for (std::uint64_t index = 0U; index < output.layout().elements();
       ++index) {
    const std::size_t offset = static_cast<std::size_t>(index);
    output_data[offset] = left_data[offset] + right_data[offset];
  }
}

void execute_add_broadcast(
    const ExecutionStep& step, const std::span<const PlanValue> values,
    const std::span<const float* const> value_data,
    float* const output_data) noexcept {
  assert(step.operands().size() == 2U);
  const PlanValue& output = value_at(values, step.output());
  const PlanValue& left = value_at(values, step.operands()[0]);
  const PlanValue& right = value_at(values, step.operands()[1]);
  const float* const left_data = data_at(value_data, left.source_value());
  const float* const right_data = data_at(value_data, right.source_value());

  for (std::uint64_t index = 0U; index < output.layout().elements();
       ++index) {
    const auto coordinates = unravel(index, output);
    const std::size_t left_offset = static_cast<std::size_t>(
        broadcast_offset(left, output, coordinates));
    const std::size_t right_offset = static_cast<std::size_t>(
        broadcast_offset(right, output, coordinates));
    output_data[static_cast<std::size_t>(index)] =
        left_data[left_offset] + right_data[right_offset];
  }
}

void execute_matmul_rank2(
    const ExecutionStep& step, const std::span<const PlanValue> values,
    const std::span<const float* const> value_data,
    float* const output_data) noexcept {
  assert(step.operands().size() == 2U);
  [[maybe_unused]] const PlanValue& output =
      value_at(values, step.output());
  const PlanValue& left = value_at(values, step.operands()[0]);
  const PlanValue& right = value_at(values, step.operands()[1]);
  assert(left.layout().rank() == 2U);
  assert(right.layout().rank() == 2U);
  assert(output.layout().rank() == 2U);
  const float* const left_data = data_at(value_data, left.source_value());
  const float* const right_data = data_at(value_data, right.source_value());

  const std::uint64_t rows = extent(left, 0U);
  const std::uint64_t inner = extent(left, 1U);
  const std::uint64_t columns = extent(right, 1U);
  assert(inner == extent(right, 0U));
  for (std::uint64_t row = 0U; row < rows; ++row) {
    for (std::uint64_t column = 0U; column < columns; ++column) {
      double accumulator = 0.0;
      for (std::uint64_t reduction = 0U; reduction < inner; ++reduction) {
        const std::size_t left_offset = static_cast<std::size_t>(
            row * inner + reduction);
        const std::size_t right_offset = static_cast<std::size_t>(
            reduction * columns + column);
        accumulator = round_to_binary64(
            accumulator + static_cast<double>(left_data[left_offset]) *
                              static_cast<double>(right_data[right_offset]));
      }
      output_data[static_cast<std::size_t>(row * columns + column)] =
          static_cast<float>(accumulator);
    }
  }
}

void execute_matmul_batched(
    const ExecutionStep& step, const std::span<const PlanValue> values,
    const std::span<const float* const> value_data,
    float* const output_data) noexcept {
  assert(step.operands().size() == 2U);
  const PlanValue& output = value_at(values, step.output());
  const PlanValue& left = value_at(values, step.operands()[0]);
  const PlanValue& right = value_at(values, step.operands()[1]);
  const float* const left_data = data_at(value_data, left.source_value());
  const float* const right_data = data_at(value_data, right.source_value());

  const std::size_t left_row_axis = left.layout().rank() - 2U;
  const std::size_t left_inner_axis = left.layout().rank() - 1U;
  const std::size_t right_inner_axis = right.layout().rank() - 2U;
  const std::size_t right_column_axis = right.layout().rank() - 1U;
  const std::size_t output_row_axis = output.layout().rank() - 2U;
  const std::size_t output_column_axis = output.layout().rank() - 1U;
  const std::uint64_t inner = extent(left, left_inner_axis);
  assert(inner == extent(right, right_inner_axis));
  const auto left_strides = left.layout().strides_elements();
  const auto right_strides = right.layout().strides_elements();

  for (std::uint64_t index = 0U; index < output.layout().elements();
       ++index) {
    const auto coordinates = unravel(index, output);
    const std::uint64_t row = coordinates[output_row_axis];
    const std::uint64_t column = coordinates[output_column_axis];
    const std::uint64_t left_batch =
        matrix_batch_offset(left, output, coordinates);
    const std::uint64_t right_batch =
        matrix_batch_offset(right, output, coordinates);

    double accumulator = 0.0;
    for (std::uint64_t reduction = 0U; reduction < inner; ++reduction) {
      const std::uint64_t left_offset =
          left_batch + row * left_strides[left_row_axis] +
          reduction * left_strides[left_inner_axis];
      const std::uint64_t right_offset =
          right_batch + reduction * right_strides[right_inner_axis] +
          column * right_strides[right_column_axis];
      assert(left_offset < left.layout().elements());
      assert(right_offset < right.layout().elements());
      accumulator = round_to_binary64(
          accumulator +
          static_cast<double>(
              left_data[static_cast<std::size_t>(left_offset)]) *
              static_cast<double>(
                  right_data[static_cast<std::size_t>(right_offset)]));
    }
    output_data[static_cast<std::size_t>(index)] =
        static_cast<float>(accumulator);
  }
}

void execute_relu(const ExecutionStep& step,
                  const std::span<const PlanValue> values,
                  const std::span<const float* const> value_data,
                  float* const output_data) noexcept {
  assert(step.operands().size() == 1U);
  const PlanValue& output = value_at(values, step.output());
  const float* const input_data = data_at(value_data, step.operands()[0]);
  for (std::uint64_t index = 0U; index < output.layout().elements();
       ++index) {
    const std::size_t offset = static_cast<std::size_t>(index);
    const float value = input_data[offset];
    output_data[offset] =
        std::isnan(value) ? value : (value > 0.0F ? value : 0.0F);
  }
}

}  // namespace

void execute_dense_kernel(
    const ExecutionStep& step, const std::span<const PlanValue> values,
    const std::span<const float* const> value_data,
    float* const output) noexcept {
  assert(output != nullptr);
  switch (step.kernel()) {
    case DenseKernelKind::add_contiguous_f32:
      execute_add_contiguous(step, values, value_data, output);
      return;
    case DenseKernelKind::add_broadcast_f32:
      execute_add_broadcast(step, values, value_data, output);
      return;
    case DenseKernelKind::matmul_rank2_f32:
      execute_matmul_rank2(step, values, value_data, output);
      return;
    case DenseKernelKind::matmul_batched_f32:
      execute_matmul_batched(step, values, value_data, output);
      return;
    case DenseKernelKind::relu_contiguous_f32:
      execute_relu(step, values, value_data, output);
      return;
  }
}

}  // namespace tensorkiln::detail
