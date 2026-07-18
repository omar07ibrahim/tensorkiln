#include "test.hpp"

#include <array>
#include <bit>
#include <cfenv>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#if defined(__SSE__) && \
    (defined(__i386__) || defined(__x86_64__))
#include <xmmintrin.h>
#endif

#include "tensorkiln/execution.hpp"
#include "tensorkiln/reference.hpp"

#include "../src/execution_internal.hpp"

namespace {

using tensorkiln::Diagnostic;
using tensorkiln::ErrorCode;
using tensorkiln::ExecutionInputBinding;
using tensorkiln::ExecutionPlan;
using tensorkiln::ExecutionPlanCompiler;
using tensorkiln::ExecutionResultView;
using tensorkiln::ExecutionRunStatus;
using tensorkiln::ExecutionSession;
using tensorkiln::GraphBuilder;
using tensorkiln::InputBinding;
using tensorkiln::ReferenceInterpreter;
using tensorkiln::ReferenceResult;
using tensorkiln::Shape;
using tensorkiln::TensorType;
using tensorkiln::TensorView;
using tensorkiln::ValueId;
using tensorkiln::VerifiedGraph;

#if defined(__i386__) && (defined(__GNUC__) || defined(__clang__)) && \
    defined(__FLT_EVAL_METHOD__) && __FLT_EVAL_METHOD__ == 2
class X87PrecisionGuard final {
 public:
  X87PrecisionGuard() noexcept {
    __asm__ volatile("fnstcw %0" : "=m"(original_));
    const std::uint16_t single_precision =
        static_cast<std::uint16_t>(original_ & UINT16_C(0xfcff));
    __asm__ volatile("fldcw %0" : : "m"(single_precision));
  }

  X87PrecisionGuard(const X87PrecisionGuard&) = delete;
  X87PrecisionGuard& operator=(const X87PrecisionGuard&) = delete;

  ~X87PrecisionGuard() {
    __asm__ volatile("fldcw %0" : : "m"(original_));
  }

 private:
  std::uint16_t original_ = 0U;
};
#endif

#if defined(__SSE__) && \
    (defined(__i386__) || defined(__x86_64__))
class MxcsrGuard final {
 public:
  explicit MxcsrGuard(const unsigned int rounding_mode) noexcept
      : original_(_mm_getcsr()) {
    _mm_setcsr((original_ & ~static_cast<unsigned int>(_MM_ROUND_MASK)) |
               rounding_mode);
  }

  MxcsrGuard(const MxcsrGuard&) = delete;
  MxcsrGuard& operator=(const MxcsrGuard&) = delete;

  ~MxcsrGuard() { _mm_setcsr(original_); }

 private:
  unsigned int original_;
};
#endif

template <typename T>
[[nodiscard]] T unwrap(tensorkiln::Result<T> result) {
  TK_REQUIRE(result.value_if() != nullptr);
  return std::move(*result.value_if());
}

template <typename T>
const Diagnostic& require_error(
    const tensorkiln::Result<T>& result, const ErrorCode code) {
  TK_REQUIRE(result.error_if() != nullptr);
  TK_REQUIRE_EQ(result.error_if()->code, code);
  return *result.error_if();
}

[[nodiscard]] TensorType f32(
    const std::initializer_list<std::int64_t> extents) {
  return unwrap(TensorType::create(unwrap(Shape::create(extents))));
}

void require_bit_equal(const std::span<const float> actual,
                       const std::span<const float> expected) {
  TK_REQUIRE_EQ(actual.size(), expected.size());
  for (std::size_t index = 0U; index < actual.size(); ++index) {
    TK_REQUIRE_EQ(std::bit_cast<std::uint32_t>(actual[index]),
                  std::bit_cast<std::uint32_t>(expected[index]));
  }
}

[[nodiscard]] TensorView require_output(const ExecutionResultView& result,
                                        const std::string_view name) {
  const auto output = result.output(name);
  TK_REQUIRE(output.has_value());
  return *output;
}

[[nodiscard]] const tensorkiln::Tensor& require_reference_output(
    const ReferenceResult& result, const std::string_view name) {
  const tensorkiln::Tensor* output = result.output(name);
  TK_REQUIRE(output != nullptr);
  return *output;
}

struct ExecutableFixture final {
  VerifiedGraph graph;
  ValueId input;
};

[[nodiscard]] ExecutableFixture make_executable_graph() {
  GraphBuilder builder;
  const ValueId input = unwrap(builder.input("x", f32({2, 3})));
  const std::array<float, 6U> weight_data{{
      1.0F, -2.0F,
      3.0F, 4.0F,
      -1.0F, 2.0F,
  }};
  const ValueId weight =
      unwrap(builder.constant("weight", f32({3, 2}), weight_data));
  const ValueId product = unwrap(builder.matmul(input, weight));
  const std::array<float, 2U> bias_data{{0.5F, -1.0F}};
  const ValueId bias =
      unwrap(builder.constant("bias", f32({2}), bias_data));
  const ValueId shifted = unwrap(builder.add(product, bias));
  const ValueId result = unwrap(builder.relu(shifted));
  static_cast<void>(unwrap(builder.output("result", result)));
  static_cast<void>(unwrap(builder.output("result_alias", result)));
  static_cast<void>(unwrap(builder.output("raw_product", product)));
  return ExecutableFixture{unwrap(std::move(builder).finish()), input};
}

TK_TEST("Execution session matches the independent reference pipeline") {
  const ExecutableFixture fixture = make_executable_graph();
  const ExecutionPlan plan = unwrap(ExecutionPlanCompiler::run(fixture.graph));
  ExecutionSession session = ExecutionSession::create(plan);
  const std::array<float, 6U> input{{
      1.0F, 2.0F, 3.0F,
      -1.0F, 0.5F, 4.0F,
  }};
  const std::array<ExecutionInputBinding, 1U> bindings{{{"x", input}}};
  const auto bound = session.bind(bindings);
  TK_REQUIRE(bound.has_value());
  TK_REQUIRE_EQ(bound.value_if()->input_count(), 1U);
  TK_REQUIRE_EQ(session.workspace_bytes(), plan.stats().workspace_bytes);
  TK_REQUIRE_EQ(session.run(), ExecutionRunStatus::success);

  const auto result = session.result();
  TK_REQUIRE(result.has_value());
  TK_REQUIRE(result->current());
  const TensorView actual = require_output(*result, "result");
  const TensorView alias = require_output(*result, "result_alias");
  const TensorView raw = require_output(*result, "raw_product");
  TK_REQUIRE(actual.data().data() == alias.data().data());

  const std::array<InputBinding, 1U> reference_bindings{{{"x", input}}};
  const ReferenceResult reference =
      unwrap(ReferenceInterpreter::run(fixture.graph, reference_bindings));
  require_bit_equal(actual.data(),
                    require_reference_output(reference, "result").data());
  require_bit_equal(raw.data(),
                    require_reference_output(reference, "raw_product").data());
}

TK_TEST("Execution binding diagnostics preserve fail-closed precedence") {
  GraphBuilder builder;
  const ValueId left = unwrap(builder.input("left", f32({2})));
  const ValueId right = unwrap(builder.input("right", f32({2})));
  const ValueId sum = unwrap(builder.add(left, right));
  static_cast<void>(unwrap(builder.output("sum", sum)));
  const VerifiedGraph graph = unwrap(std::move(builder).finish());
  const ExecutionPlan plan = unwrap(ExecutionPlanCompiler::run(graph));
  ExecutionSession session = ExecutionSession::create(plan);
  TK_REQUIRE_EQ(session.run(), ExecutionRunStatus::inputs_not_bound);

  const std::array<float, 2U> values{{1.0F, 2.0F}};
  const std::array<ExecutionInputBinding, 3U> too_many{{
      {"left", values},
      {"right", values},
      {"unknown", values},
  }};
  require_error(session.bind(too_many),
                ErrorCode::input_binding_count_exceeded);

  const std::array<ExecutionInputBinding, 2U> unknown{{
      {"left", values},
      {"unknown", values},
  }};
  require_error(session.bind(unknown), ErrorCode::input_binding_unknown);

  const std::array<ExecutionInputBinding, 2U> duplicate{{
      {"left", values},
      {"left", values},
  }};
  require_error(session.bind(duplicate), ErrorCode::input_binding_duplicate);

  const std::array<ExecutionInputBinding, 1U> missing{{{"left", values}}};
  require_error(session.bind(missing), ErrorCode::input_binding_missing);

  const std::array<float, 1U> short_values{{1.0F}};
  const std::array<ExecutionInputBinding, 2U> wrong_size{{
      {"left", short_values},
      {"right", values},
  }};
  require_error(session.bind(wrong_size),
                ErrorCode::input_binding_size_mismatch);
  TK_REQUIRE_EQ(session.run(), ExecutionRunStatus::inputs_not_bound);
}

TK_TEST("External outputs alias borrowed inputs and plan-owned constants") {
  GraphBuilder builder;
  const ValueId input = unwrap(builder.input("x", f32({2})));
  const std::array<float, 2U> constant_data{{3.0F, -4.0F}};
  const ValueId constant =
      unwrap(builder.constant("c", f32({2}), constant_data));
  static_cast<void>(unwrap(builder.output("input", input)));
  static_cast<void>(unwrap(builder.output("input_alias", input)));
  static_cast<void>(unwrap(builder.output("constant", constant)));
  const VerifiedGraph graph = unwrap(std::move(builder).finish());
  const ExecutionPlan plan = unwrap(ExecutionPlanCompiler::run(graph));
  TK_REQUIRE(plan.steps().empty());
  TK_REQUIRE_EQ(plan.stats().workspace_bytes, 0U);

  ExecutionSession session = ExecutionSession::create(plan);
  const std::array<float, 2U> input_data{{8.0F, 9.0F}};
  const std::array<ExecutionInputBinding, 1U> bindings{{{"x", input_data}}};
  TK_REQUIRE(session.bind(bindings).has_value());
  TK_REQUIRE_EQ(session.run(), ExecutionRunStatus::success);
  const ExecutionResultView result = *session.result();
  const TensorView input_output = require_output(result, "input");
  const TensorView input_alias = require_output(result, "input_alias");
  const TensorView constant_output = require_output(result, "constant");
  TK_REQUIRE(input_output.data().data() == input_data.data());
  TK_REQUIRE(input_output.data().data() == input_alias.data().data());
  require_bit_equal(constant_output.data(), constant_data);

  ExecutionSession moved(std::move(session));
  TK_REQUIRE(!result.current());
  TK_REQUIRE(!result.output("input").has_value());
  TK_REQUIRE(!moved.result().has_value());
}

TK_TEST("Result views become safely stale after session destruction") {
  const ExecutableFixture fixture = make_executable_graph();
  const ExecutionPlan plan = unwrap(ExecutionPlanCompiler::run(fixture.graph));
  const std::array<float, 6U> input{};
  const std::array<ExecutionInputBinding, 1U> bindings{{{"x", input}}};
  std::optional<ExecutionResultView> escaped;
  {
    ExecutionSession session = ExecutionSession::create(plan);
    TK_REQUIRE(session.bind(bindings).has_value());
    TK_REQUIRE_EQ(session.run(), ExecutionRunStatus::success);
    escaped = session.result();
    TK_REQUIRE(escaped.has_value());
    TK_REQUIRE(escaped->current());
  }
  TK_REQUIRE(escaped.has_value());
  TK_REQUIRE(!escaped->current());
  TK_REQUIRE(!escaped->output("result").has_value());
}

TK_TEST("Execution rejects exact and partial aliases into its own arena") {
  GraphBuilder builder;
  const ValueId input = unwrap(builder.input("x", f32({2})));
  const ValueId seed = unwrap(builder.input("seed", f32({4})));
  const ValueId retained = unwrap(builder.relu(seed));
  const ValueId result = unwrap(builder.relu(input));
  static_cast<void>(unwrap(builder.output("retained", retained)));
  static_cast<void>(unwrap(builder.output("result", result)));
  const VerifiedGraph graph = unwrap(std::move(builder).finish());
  const ExecutionPlan plan = unwrap(ExecutionPlanCompiler::run(graph));
  ExecutionSession session = ExecutionSession::create(plan);
  const std::array<float, 2U> input_data{{1.0F, 2.0F}};
  const std::array<float, 4U> first_seed{{1.0F, 2.0F, 3.0F, 4.0F}};
  const std::array<ExecutionInputBinding, 2U> first_bindings{{
      {"x", input_data},
      {"seed", first_seed},
  }};
  TK_REQUIRE(session.bind(first_bindings).has_value());
  TK_REQUIRE_EQ(session.run(), ExecutionRunStatus::success);
  const TensorView retained_output =
      require_output(*session.result(), "retained");
  TK_REQUIRE_EQ(retained_output.data().size(), 4U);

  const std::array<float, 4U> second_seed{{5.0F, 6.0F, 7.0F, 8.0F}};
  const std::span<const float> exact_alias{retained_output.data().data(), 2U};
  const std::array<ExecutionInputBinding, 2U> exact_bindings{{
      {"x", exact_alias},
      {"seed", second_seed},
  }};
  require_error(session.bind(exact_bindings),
                ErrorCode::input_binding_aliases_workspace);

  const std::span<const float> partial_alias{
      retained_output.data().data() + 1U, 2U};
  const std::array<ExecutionInputBinding, 2U> partial_bindings{{
      {"x", partial_alias},
      {"seed", second_seed},
  }};
  require_error(session.bind(partial_bindings),
                ErrorCode::input_binding_aliases_workspace);
  TK_REQUIRE_EQ(session.run(), ExecutionRunStatus::inputs_not_bound);
  TK_REQUIRE_EQ(tensorkiln::error_code_name(
                    ErrorCode::input_binding_aliases_workspace),
                "input_binding_aliases_workspace");
}

TK_TEST("Repeated bindings invalidate stale views and do not leak arena state") {
  const ExecutableFixture fixture = make_executable_graph();
  const ExecutionPlan plan = unwrap(ExecutionPlanCompiler::run(fixture.graph));
  ExecutionSession session = ExecutionSession::create(plan);
  const std::array<float, 6U> first{{1.0F, 2.0F, 3.0F,
                                    4.0F, 5.0F, 6.0F}};
  const std::array<float, 6U> second{{-1.0F, -2.0F, -3.0F,
                                     0.25F, 0.5F, 0.75F}};

  const std::array<ExecutionInputBinding, 1U> first_binding{{{"x", first}}};
  TK_REQUIRE(session.bind(first_binding).has_value());
  TK_REQUIRE_EQ(session.run(), ExecutionRunStatus::success);
  const ExecutionResultView first_view = *session.result();
  const std::vector<float> first_result(
      require_output(first_view, "result").data().begin(),
      require_output(first_view, "result").data().end());

  const std::array<ExecutionInputBinding, 1U> second_binding{{{"x", second}}};
  TK_REQUIRE(session.bind(second_binding).has_value());
  TK_REQUIRE(!first_view.current());
  TK_REQUIRE_EQ(session.run(), ExecutionRunStatus::success);

  TK_REQUIRE(session.bind(first_binding).has_value());
  TK_REQUIRE_EQ(session.run(), ExecutionRunStatus::success);
  require_bit_equal(require_output(*session.result(), "result").data(),
                    first_result);
}

TK_TEST("A successful binding remains active across repeated runs") {
  GraphBuilder builder;
  const ValueId input = unwrap(builder.input("x", f32({2})));
  const ValueId result = unwrap(builder.relu(input));
  static_cast<void>(unwrap(builder.output("result", result)));
  const VerifiedGraph graph = unwrap(std::move(builder).finish());
  const ExecutionPlan plan = unwrap(ExecutionPlanCompiler::run(graph));
  ExecutionSession session = ExecutionSession::create(plan);
  const std::array<float, 2U> values{{-1.0F, 3.0F}};
  const std::array<ExecutionInputBinding, 1U> bindings{{{"x", values}}};
  TK_REQUIRE(session.bind(bindings).has_value());

  TK_REQUIRE_EQ(session.run(), ExecutionRunStatus::success);
  require_bit_equal(require_output(*session.result(), "result").data(),
                    std::array<float, 2U>{{0.0F, 3.0F}});
  TK_REQUIRE_EQ(session.run(), ExecutionRunStatus::success);
  require_bit_equal(require_output(*session.result(), "result").data(),
                    std::array<float, 2U>{{0.0F, 3.0F}});
}

TK_TEST("Batched broadcast MatMul is bit-exact with the reference") {
  GraphBuilder builder;
  const ValueId left = unwrap(builder.input("left", f32({2, 1, 2, 3})));
  const ValueId right = unwrap(builder.input("right", f32({1, 4, 3, 2})));
  const ValueId product = unwrap(builder.matmul(left, right));
  static_cast<void>(unwrap(builder.output("product", product)));
  const VerifiedGraph graph = unwrap(std::move(builder).finish());
  const ExecutionPlan plan = unwrap(ExecutionPlanCompiler::run(graph));
  TK_REQUIRE_EQ(plan.steps().size(), 1U);
  TK_REQUIRE_EQ(plan.steps()[0].kernel(),
                tensorkiln::DenseKernelKind::matmul_batched_f32);

  std::array<float, 12U> left_data{};
  std::array<float, 24U> right_data{};
  for (std::size_t index = 0U; index < left_data.size(); ++index) {
    left_data[index] = static_cast<float>(index) * 0.25F - 1.0F;
  }
  for (std::size_t index = 0U; index < right_data.size(); ++index) {
    right_data[index] = static_cast<float>(index % 7U) * 0.5F - 1.5F;
  }

  ExecutionSession session = ExecutionSession::create(plan);
  const std::array<ExecutionInputBinding, 2U> bindings{{
      {"left", left_data},
      {"right", right_data},
  }};
  TK_REQUIRE(session.bind(bindings).has_value());
  TK_REQUIRE_EQ(session.run(), ExecutionRunStatus::success);
  const std::array<InputBinding, 2U> reference_bindings{{
      {"left", left_data},
      {"right", right_data},
  }};
  const ReferenceResult reference =
      unwrap(ReferenceInterpreter::run(graph, reference_bindings));
  require_bit_equal(
      require_output(*session.result(), "product").data(),
      require_reference_output(reference, "product").data());
}

TK_TEST("MatMul rounds every reduction step to binary64 on all targets") {
  const std::array<float, 3U> left_data{{
      1073741824.0F,
      1.0F,
      -1073741824.0F,
  }};
  const std::array<float, 3U> right_data{{
      1073741824.0F,
      1.0F,
      1073741824.0F,
  }};

  const auto execute = [&](const TensorType left_type,
                           const TensorType right_type) {
    GraphBuilder builder;
    const ValueId left = unwrap(builder.input("left", left_type));
    const ValueId right = unwrap(builder.input("right", right_type));
    const ValueId product = unwrap(builder.matmul(left, right));
    static_cast<void>(unwrap(builder.output("product", product)));
    const VerifiedGraph graph = unwrap(std::move(builder).finish());
    const ExecutionPlan plan = unwrap(ExecutionPlanCompiler::run(graph));
    ExecutionSession session = ExecutionSession::create(plan);
    const std::array<ExecutionInputBinding, 2U> bindings{{
        {"left", left_data},
        {"right", right_data},
    }};
    TK_REQUIRE(session.bind(bindings).has_value());
    TK_REQUIRE_EQ(session.run(), ExecutionRunStatus::success);
    const TensorView output = require_output(*session.result(), "product");
    TK_REQUIRE_EQ(output.data().size(), 1U);
    TK_REQUIRE_EQ(std::bit_cast<std::uint32_t>(output.data()[0]), 0U);

    const std::array<InputBinding, 2U> reference_bindings{{
        {"left", left_data},
        {"right", right_data},
    }};
    const ReferenceResult reference =
        unwrap(ReferenceInterpreter::run(graph, reference_bindings));
    TK_REQUIRE_EQ(
        std::bit_cast<std::uint32_t>(
            require_reference_output(reference, "product").data()[0]),
        0U);
  };

  execute(f32({1, 3}), f32({3, 1}));
  execute(f32({1, 1, 3}), f32({1, 3, 1}));
}

TK_TEST("ReLU execution preserves NaN bits and maps signed zero to positive") {
  GraphBuilder builder;
  const ValueId input = unwrap(builder.input("x", f32({6})));
  const ValueId result = unwrap(builder.relu(input));
  static_cast<void>(unwrap(builder.output("result", result)));
  const VerifiedGraph graph = unwrap(std::move(builder).finish());
  const ExecutionPlan plan = unwrap(ExecutionPlanCompiler::run(graph));
  ExecutionSession session = ExecutionSession::create(plan);
  const std::array<float, 6U> values{{
      std::bit_cast<float>(UINT32_C(0x80000000)),
      0.0F,
      -2.0F,
      3.0F,
      std::bit_cast<float>(UINT32_C(0x7fc12345)),
      std::bit_cast<float>(UINT32_C(0xffc54321)),
  }};
  const std::array<ExecutionInputBinding, 1U> bindings{{{"x", values}}};
  TK_REQUIRE(session.bind(bindings).has_value());
  TK_REQUIRE_EQ(session.run(), ExecutionRunStatus::success);
  const TensorView output = require_output(*session.result(), "result");
  TK_REQUIRE_EQ(std::bit_cast<std::uint32_t>(output.data()[0]), 0U);
  TK_REQUIRE_EQ(std::bit_cast<std::uint32_t>(output.data()[4]),
                UINT32_C(0x7fc12345));
  TK_REQUIRE_EQ(std::bit_cast<std::uint32_t>(output.data()[5]),
                UINT32_C(0xffc54321));
}

TK_TEST("Optional kernel write audit rejects an in-arena stray write") {
  GraphBuilder builder;
  const ValueId input = unwrap(builder.input("x", f32({15})));
  const ValueId result = unwrap(builder.relu(input));
  static_cast<void>(unwrap(builder.output("result", result)));
  const VerifiedGraph graph = unwrap(std::move(builder).finish());
  const ExecutionPlan plan = unwrap(ExecutionPlanCompiler::run(graph));
  TK_REQUIRE_EQ(plan.stats().workspace_bytes, 64U);

  const std::array<float, 15U> values{};
  const std::array<ExecutionInputBinding, 1U> bindings{{{"x", values}}};
  ExecutionSession corrupted = ExecutionSession::create(
      plan, tensorkiln::ExecutionSessionOptions{true});
  TK_REQUIRE(corrupted.audits_kernel_writes());
  TK_REQUIRE(corrupted.bind(bindings).has_value());
  tensorkiln::detail::set_execution_fault_for_test(
      corrupted,
      tensorkiln::detail::ExecutionFaultKind::write_outside_output);
  TK_REQUIRE_EQ(corrupted.run(), ExecutionRunStatus::memory_corruption);
  TK_REQUIRE(!corrupted.result().has_value());

  ExecutionSession intact = ExecutionSession::create(
      plan, tensorkiln::ExecutionSessionOptions{true});
  TK_REQUIRE(intact.bind(bindings).has_value());
  TK_REQUIRE_EQ(intact.run(), ExecutionRunStatus::success);
  TK_REQUIRE(intact.result().has_value());
}

TK_TEST("Execution reports unsupported rounding mode without committing") {
  const ExecutableFixture fixture = make_executable_graph();
  const ExecutionPlan plan = unwrap(ExecutionPlanCompiler::run(fixture.graph));
  ExecutionSession session = ExecutionSession::create(plan);
  const std::array<float, 6U> input{};
  const std::array<ExecutionInputBinding, 1U> bindings{{{"x", input}}};
  TK_REQUIRE(session.bind(bindings).has_value());

  const int previous = std::fegetround();
  TK_REQUIRE(previous != -1);
  TK_REQUIRE(std::fesetround(FE_DOWNWARD) == 0);
  const ExecutionRunStatus status = session.run();
  TK_REQUIRE_EQ(std::fegetround(), FE_DOWNWARD);
  TK_REQUIRE(std::fesetround(previous) == 0);
  TK_REQUIRE_EQ(status, ExecutionRunStatus::unsupported_rounding_mode);
  TK_REQUIRE(!session.result().has_value());
  TK_REQUIRE_EQ(tensorkiln::execution_run_status_name(status),
                "unsupported_rounding_mode");
}

#if defined(__SSE__) && \
    (defined(__i386__) || defined(__x86_64__))
TK_TEST("Execution rejects a desynchronized SSE rounding mode") {
  GraphBuilder builder;
  const ValueId left = unwrap(builder.input("left", f32({1})));
  const ValueId right = unwrap(builder.input("right", f32({1})));
  const ValueId sum = unwrap(builder.add(left, right));
  static_cast<void>(unwrap(builder.output("sum", sum)));
  const VerifiedGraph graph = unwrap(std::move(builder).finish());
  const ExecutionPlan plan = unwrap(ExecutionPlanCompiler::run(graph));
  ExecutionSession session = ExecutionSession::create(plan);
  const std::array<float, 1U> left_data{{1.0F}};
  const std::array<float, 1U> right_data{{0x1p-24F}};
  const std::array<ExecutionInputBinding, 2U> bindings{{
      {"left", left_data},
      {"right", right_data},
  }};
  const std::array<InputBinding, 2U> reference_bindings{{
      {"left", left_data},
      {"right", right_data},
  }};
  TK_REQUIRE(session.bind(bindings).has_value());

  {
    const MxcsrGuard upward(_MM_ROUND_UP);
    TK_REQUIRE_EQ(std::fegetround(), FE_TONEAREST);
    TK_REQUIRE_EQ(session.run(),
                  ExecutionRunStatus::unsupported_rounding_mode);
    require_error(ReferenceInterpreter::run(graph, reference_bindings),
                  ErrorCode::unsupported_rounding_mode);
  }

  TK_REQUIRE_EQ(session.run(), ExecutionRunStatus::success);
  const TensorView output = require_output(*session.result(), "sum");
  TK_REQUIRE_EQ(std::bit_cast<std::uint32_t>(output.data()[0]),
                UINT32_C(0x3f800000));
}
#endif

#if defined(__i386__) && (defined(__GNUC__) || defined(__clang__)) && \
    defined(__FLT_EVAL_METHOD__) && __FLT_EVAL_METHOD__ == 2
TK_TEST("Execution rejects an x87 single-precision control word") {
  GraphBuilder builder;
  const ValueId left = unwrap(builder.input("left", f32({1, 3})));
  const ValueId right = unwrap(builder.input("right", f32({3, 1})));
  const ValueId product = unwrap(builder.matmul(left, right));
  static_cast<void>(unwrap(builder.output("product", product)));
  const VerifiedGraph graph = unwrap(std::move(builder).finish());
  const ExecutionPlan plan = unwrap(ExecutionPlanCompiler::run(graph));
  ExecutionSession session = ExecutionSession::create(plan);
  const std::array<float, 3U> left_data{{32768.0F, 1.0F, -32768.0F}};
  const std::array<float, 3U> right_data{{32768.0F, 1.0F, 32768.0F}};
  const std::array<ExecutionInputBinding, 2U> bindings{{
      {"left", left_data},
      {"right", right_data},
  }};
  const std::array<InputBinding, 2U> reference_bindings{{
      {"left", left_data},
      {"right", right_data},
  }};
  TK_REQUIRE(session.bind(bindings).has_value());

  {
    const X87PrecisionGuard single_precision;
    TK_REQUIRE_EQ(std::fegetround(), FE_TONEAREST);
    TK_REQUIRE_EQ(session.run(),
                  ExecutionRunStatus::unsupported_binary64_precision);
    require_error(ReferenceInterpreter::run(graph, reference_bindings),
                  ErrorCode::unsupported_binary64_precision);
  }

  TK_REQUIRE_EQ(session.run(), ExecutionRunStatus::success);
  const TensorView output = require_output(*session.result(), "product");
  TK_REQUIRE_EQ(std::bit_cast<std::uint32_t>(output.data()[0]),
                std::bit_cast<std::uint32_t>(1.0F));
  TK_REQUIRE_EQ(tensorkiln::execution_run_status_name(
                    ExecutionRunStatus::unsupported_binary64_precision),
                "unsupported_binary64_precision");
}
#endif

}  // namespace
