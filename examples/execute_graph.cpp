#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <initializer_list>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "tensorkiln/execution.hpp"
#include "tensorkiln/reference.hpp"

namespace {

template <typename T>
[[nodiscard]] T unwrap(tensorkiln::Result<T> result) {
  if (result.value_if() == nullptr) {
    const tensorkiln::Diagnostic& error = *result.error_if();
    throw std::runtime_error(
        std::string(tensorkiln::error_code_name(error.code)) + ": " +
        error.message);
  }
  return std::move(*result.value_if());
}

[[nodiscard]] tensorkiln::TensorType f32(
    const std::initializer_list<std::int64_t> extents) {
  tensorkiln::Shape shape = unwrap(tensorkiln::Shape::create(extents));
  return unwrap(tensorkiln::TensorType::create(std::move(shape)));
}

[[nodiscard]] const tensorkiln::Tensor& require_reference_output(
    const tensorkiln::ReferenceResult& result,
    const std::string_view name) {
  const tensorkiln::Tensor* output = result.output(name);
  if (output == nullptr) {
    throw std::runtime_error(
        "reference result has no @" + std::string(name) + " output");
  }
  return *output;
}

[[nodiscard]] tensorkiln::TensorView require_execution_output(
    const tensorkiln::ExecutionResultView& result,
    const std::string_view name) {
  const auto output = result.output(name);
  if (!output.has_value()) {
    throw std::runtime_error(
        "execution result has no current @" + std::string(name) +
        " output");
  }
  return *output;
}

void require_same_bits(const std::span<const float> actual,
                       const std::span<const float> expected,
                       const std::string_view context) {
  if (actual.size() != expected.size()) {
    throw std::runtime_error(std::string(context) + " size differs");
  }
  for (std::size_t index = 0U; index < actual.size(); ++index) {
    if (std::bit_cast<std::uint32_t>(actual[index]) !=
        std::bit_cast<std::uint32_t>(expected[index])) {
      throw std::runtime_error(
          std::string(context) + " differs at element " +
          std::to_string(index));
    }
  }
}

void require_expected_plan(const tensorkiln::ExecutionPlan& plan) {
  const tensorkiln::ArenaPlanStats& arena_stats =
      plan.arena_plan().stats();
  if (plan.steps().size() != 3U ||
      plan.steps()[0].kernel() !=
          tensorkiln::DenseKernelKind::matmul_rank2_f32 ||
      plan.steps()[1].kernel() !=
          tensorkiln::DenseKernelKind::add_broadcast_f32 ||
      plan.steps()[2].kernel() !=
          tensorkiln::DenseKernelKind::relu_contiguous_f32 ||
      arena_stats.buffer_count != 3U ||
      arena_stats.total_reserved_bytes != 192U ||
      arena_stats.workspace_bytes != 128U ||
      plan.stats().workspace_bytes != 128U) {
    throw std::runtime_error(
        "execution planner changed the documented three-kernel layout");
  }
}

void print_values(const std::span<const float> values) {
  std::cout << "result = [";
  for (std::size_t index = 0U; index < values.size(); ++index) {
    std::cout << (index == 0U ? "" : ", ") << values[index];
  }
  std::cout << "]\n";
}

}  // namespace

int main() {
  try {
    tensorkiln::GraphBuilder builder;
    const tensorkiln::ValueId input =
        unwrap(builder.input("x", f32({2, 3})));
    constexpr std::array<float, 6U> weight_data{{
        1.0F, -2.0F,
        3.0F, 4.0F,
        -1.0F, 2.0F,
    }};
    const tensorkiln::ValueId weight =
        unwrap(builder.constant("weight", f32({3, 2}), weight_data));
    const tensorkiln::ValueId product =
        unwrap(builder.matmul(input, weight));
    constexpr std::array<float, 2U> bias_data{{0.5F, -1.0F}};
    const tensorkiln::ValueId bias =
        unwrap(builder.constant("bias", f32({2}), bias_data));
    const tensorkiln::ValueId shifted = unwrap(builder.add(product, bias));
    const tensorkiln::ValueId activated = unwrap(builder.relu(shifted));
    static_cast<void>(unwrap(builder.output("result", activated)));
    const tensorkiln::VerifiedGraph graph =
        unwrap(std::move(builder).finish());

    const tensorkiln::ExecutionPlan plan =
        unwrap(tensorkiln::ExecutionPlanCompiler::run(graph));
    require_expected_plan(plan);
    std::cout << "=== verified dense execution plan ===\n" << plan.dump();

    constexpr std::array<float, 6U> input_data{{
        1.0F, 2.0F, 3.0F,
        -1.0F, 0.5F, 4.0F,
    }};
    const std::array<tensorkiln::ExecutionInputBinding, 1U> bindings{{
        {"x", input_data},
    }};
    tensorkiln::ExecutionSession session = tensorkiln::ExecutionSession::create(
        plan, tensorkiln::ExecutionSessionOptions{true});
    if (!session.audits_kernel_writes()) {
      throw std::runtime_error("kernel write auditing was not enabled");
    }
    static_cast<void>(unwrap(session.bind(bindings)));
    const tensorkiln::ExecutionRunStatus run_status = session.run();
    if (run_status != tensorkiln::ExecutionRunStatus::success) {
      throw std::runtime_error(
          "execution stopped with " +
          std::string(tensorkiln::execution_run_status_name(run_status)));
    }

    const auto result = session.result();
    if (!result.has_value() || !result->current()) {
      throw std::runtime_error("execution did not publish a current result");
    }
    const tensorkiln::TensorView actual =
        require_execution_output(*result, "result");

    const std::array<tensorkiln::InputBinding, 1U> reference_bindings{{
        {"x", input_data},
    }};
    const tensorkiln::ReferenceResult reference = unwrap(
        tensorkiln::ReferenceInterpreter::run(graph, reference_bindings));
    const tensorkiln::Tensor& expected =
        require_reference_output(reference, "result");
    require_same_bits(actual.data(), expected.data(),
                      "executor and reference output");
    constexpr std::array<float, 4U> expected_values{{
        4.5F, 11.0F, 0.0F, 11.0F,
    }};
    require_same_bits(actual.data(), expected_values,
                      "documented execution output");

    print_values(actual.data());
    std::cout << "verified: audited execution matches the independent "
                 "reference bit for bit\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "TensorKiln execution example failed: " << error.what()
              << '\n';
    return EXIT_FAILURE;
  }
}
