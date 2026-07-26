#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>

#include "tensorkiln/execution.hpp"
#include "tensorkiln/reference.hpp"

namespace {

constexpr std::size_t kSliceExtent = 4U;
constexpr std::size_t kSliceCount = 5U;
constexpr std::size_t kElementCount = kSliceExtent * kSliceCount;
constexpr std::uint64_t kScalarSteps = 3U * kElementCount;
constexpr std::uint64_t kReferenceScalarSteps =
    kElementCount + kScalarSteps;

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

[[nodiscard]] tensorkiln::VerifiedGraph make_softmax_graph(
    const std::int64_t axis) {
  tensorkiln::GraphBuilder builder;
  const tensorkiln::ValueId input =
      unwrap(builder.input("x", f32({5, 4})));
  const tensorkiln::ValueId probabilities =
      unwrap(builder.softmax(input, axis));
  static_cast<void>(
      unwrap(builder.output("probabilities", probabilities)));
  return unwrap(std::move(builder).finish());
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

void require_expected_bits(
    const std::span<const float> actual,
    const std::span<const std::uint32_t> expected) {
  if (actual.size() != expected.size()) {
    throw std::runtime_error("documented Softmax fixture size differs");
  }
  for (std::size_t index = 0U; index < actual.size(); ++index) {
    if (std::bit_cast<std::uint32_t>(actual[index]) != expected[index]) {
      throw std::runtime_error(
          "documented Softmax fixture differs at element " +
          std::to_string(index));
    }
  }
}

void require_last_axis_plan(const tensorkiln::VerifiedGraph& graph,
                            const tensorkiln::ExecutionPlan& plan) {
  if (graph.nodes().size() != 2U) {
    throw std::runtime_error(
        "Softmax example graph changed its documented node count");
  }
  const auto* operation =
      std::get_if<tensorkiln::SoftmaxOp>(
          &graph.nodes()[1U].operation());
  if (operation == nullptr || operation->axis != 1U) {
    throw std::runtime_error(
        "Softmax example did not retain canonical last axis 1");
  }

  const tensorkiln::ExecutionPlanStats& stats = plan.stats();
  const tensorkiln::ArenaPlanStats& arena_stats =
      plan.arena_plan().stats();
  if (plan.steps().size() != 1U ||
      plan.steps()[0U].source_node().ordinal() != 1U ||
      plan.steps()[0U].kernel() !=
          tensorkiln::DenseKernelKind::softmax_last_axis_f32 ||
      plan.steps()[0U].scalar_steps() != kScalarSteps ||
      stats.value_count != 2U || stats.step_count != 1U ||
      stats.scalar_steps != kScalarSteps ||
      stats.workspace_bytes != 128U ||
      arena_stats.buffer_count != 1U ||
      arena_stats.total_payload_bytes != 80U ||
      arena_stats.total_reserved_bytes != 128U ||
      arena_stats.workspace_bytes != 128U) {
    throw std::runtime_error(
        "Softmax planner changed the documented one-kernel layout");
  }
}

[[nodiscard]] std::string format_f32_bits(const float value) {
  constexpr std::string_view digits{"0123456789abcdef"};
  const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  std::string result{"0x00000000"};
  for (std::size_t index = 0U; index < 8U; ++index) {
    const auto shift =
        static_cast<unsigned int>((7U - index) * 4U);
    const std::uint32_t nibble =
        (bits >> shift) & UINT32_C(0x0f);
    result[2U + index] =
        digits[static_cast<std::size_t>(nibble)];
  }
  return result;
}

void append_slice(std::string& report, const std::string_view label,
                  const std::span<const float> values) {
  if (values.size() != kSliceExtent) {
    throw std::runtime_error(
        "Softmax report received an incomplete policy slice");
  }
  report += "slice ";
  report += label;
  report += " bits=[";
  for (std::size_t index = 0U; index < values.size(); ++index) {
    if (index != 0U) {
      report += ", ";
    }
    report += format_f32_bits(values[index]);
  }
  report += "]\n";
}

}  // namespace

int main() {
  try {
    const float payload_nan =
        std::bit_cast<float>(UINT32_C(0x7fc12345));
    const float infinity = std::numeric_limits<float>::infinity();
    const std::array<float, kElementCount> input_data{{
        0.0F, 0.0F, 0.0F, 0.0F,
        payload_nan, infinity, 0.0F, -infinity,
        infinity, 3.0F, infinity, -infinity,
        -infinity, -infinity, -infinity, -infinity,
        -infinity, 0.0F, 0.0F, -infinity,
    }};
    constexpr std::array<std::uint32_t, kElementCount> expected_bits{{
        UINT32_C(0x3e800000), UINT32_C(0x3e800000),
        UINT32_C(0x3e800000), UINT32_C(0x3e800000),
        UINT32_C(0x7fc00000), UINT32_C(0x7fc00000),
        UINT32_C(0x7fc00000), UINT32_C(0x7fc00000),
        UINT32_C(0x3f000000), UINT32_C(0x00000000),
        UINT32_C(0x3f000000), UINT32_C(0x00000000),
        UINT32_C(0x7fc00000), UINT32_C(0x7fc00000),
        UINT32_C(0x7fc00000), UINT32_C(0x7fc00000),
        UINT32_C(0x00000000), UINT32_C(0x3f000000),
        UINT32_C(0x3f000000), UINT32_C(0x00000000),
    }};

    const tensorkiln::VerifiedGraph graph = make_softmax_graph(-1);
    const tensorkiln::ExecutionPlan plan =
        unwrap(tensorkiln::ExecutionPlanCompiler::run(graph));
    require_last_axis_plan(graph, plan);

    tensorkiln::ExecutionSession session =
        tensorkiln::ExecutionSession::create(
            plan, tensorkiln::ExecutionSessionOptions{true});
    if (!session.audits_kernel_writes()) {
      throw std::runtime_error("kernel write auditing was not enabled");
    }
    const std::array<tensorkiln::ExecutionInputBinding, 1U>
        execution_bindings{{{"x", input_data}}};
    static_cast<void>(unwrap(session.bind(execution_bindings)));
    const tensorkiln::ExecutionRunStatus run_status = session.run();
    if (run_status != tensorkiln::ExecutionRunStatus::success) {
      throw std::runtime_error(
          "execution stopped with " +
          std::string(
              tensorkiln::execution_run_status_name(run_status)));
    }

    const auto execution_result = session.result();
    if (!execution_result.has_value() ||
        !execution_result->current()) {
      throw std::runtime_error(
          "execution did not publish a current result");
    }
    const tensorkiln::TensorView actual =
        require_execution_output(*execution_result, "probabilities");

    const std::array<tensorkiln::InputBinding, 1U>
        reference_bindings{{{"x", input_data}}};
    const tensorkiln::ReferenceResult reference = unwrap(
        tensorkiln::ReferenceInterpreter::run(
            graph, reference_bindings));
    const tensorkiln::Tensor& expected =
        require_reference_output(reference, "probabilities");
    if (actual.type() != expected.type()) {
      throw std::runtime_error(
          "executor and reference output types differ");
    }
    require_same_bits(actual.data(), expected.data(),
                      "executor and reference output");
    require_expected_bits(actual.data(), expected_bits);

    const tensorkiln::VerifiedGraph axis_zero_graph =
        make_softmax_graph(0);
    const tensorkiln::ReferenceResult axis_zero_reference = unwrap(
        tensorkiln::ReferenceInterpreter::run(
            axis_zero_graph, reference_bindings));
    static_cast<void>(
        require_reference_output(axis_zero_reference, "probabilities"));
    if (axis_zero_reference.scalar_steps() !=
        kReferenceScalarSteps) {
      throw std::runtime_error(
          "axis-zero reference work changed from the documented bound");
    }

    const auto rejected =
        tensorkiln::ExecutionPlanCompiler::run(axis_zero_graph);
    const tensorkiln::Diagnostic* rejection = rejected.error_if();
    constexpr std::string_view expected_rejection{
        "plan backend does not support softmax axis 0 at #n1"};
    if (rejection == nullptr ||
        rejection->code !=
            tensorkiln::ErrorCode::plan_operation_unsupported ||
        rejection->message != expected_rejection) {
      throw std::runtime_error(
          "axis-zero optimized plan did not fail with the documented "
          "typed diagnostic");
    }

    std::string report{
        "=== verified Softmax execution ===\n"
        "scope: deterministic correctness example; not a benchmark\n"
        "plan {shape=f32[5,4], axis=1, "
        "kernel=softmax_last_axis_f32, scalar_steps=60, "
        "workspace_bytes=128, audited=true}\n"};
    append_slice(report, "finite_equal",
                 actual.data().subspan(0U, kSliceExtent));
    append_slice(report, "nan_precedence",
                 actual.data().subspan(4U, kSliceExtent));
    append_slice(report, "positive_infinity_split",
                 actual.data().subspan(8U, kSliceExtent));
    append_slice(report, "all_negative_infinity",
                 actual.data().subspan(12U, kSliceExtent));
    append_slice(report, "mixed_negative_infinity",
                 actual.data().subspan(16U, kSliceExtent));
    report +=
        "agreement {executor_reference_bits=20/20, "
        "executor_fixture_bits=20/20}\n"
        "=== optimized axis boundary ===\n"
        "reference_axis0 {status=accepted, scalar_steps=80}\n"
        "optimized_axis0 {code=plan_operation_unsupported, "
        "message=plan backend does not support softmax axis 0 at #n1}\n"
        "verified: last-axis execution and reference agree; valid axis 0 "
        "remains reference-only\n";

    std::cout << report;
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "TensorKiln Softmax example failed: " << error.what()
              << '\n';
    return EXIT_FAILURE;
  }
}
