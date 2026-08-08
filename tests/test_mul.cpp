#include "test.hpp"

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "tensorkiln/dead_code_elimination.hpp"
#include "tensorkiln/execution.hpp"
#include "tensorkiln/reference.hpp"
#include "tensorkiln/structural_canonicalization.hpp"

namespace {

using tensorkiln::AddOp;
using tensorkiln::ArenaAllocation;
using tensorkiln::ArenaPlacement;
using tensorkiln::DeadCodeElimination;
using tensorkiln::DeadCodeEliminationResult;
using tensorkiln::DenseKernelKind;
using tensorkiln::ErrorCode;
using tensorkiln::ExecutionInputBinding;
using tensorkiln::ExecutionPlan;
using tensorkiln::ExecutionPlanCandidate;
using tensorkiln::ExecutionPlanCompiler;
using tensorkiln::ExecutionPlanVerifier;
using tensorkiln::ExecutionRunStatus;
using tensorkiln::ExecutionSession;
using tensorkiln::ExecutionSessionOptions;
using tensorkiln::ExecutionStepSpec;
using tensorkiln::GraphArenaLowering;
using tensorkiln::GraphArenaLoweringResult;
using tensorkiln::GraphArenaPlacementVerifier;
using tensorkiln::GraphBuilder;
using tensorkiln::InputBinding;
using tensorkiln::MatMulOp;
using tensorkiln::MulOp;
using tensorkiln::Operation;
using tensorkiln::ReferenceInterpreter;
using tensorkiln::ReferenceLimits;
using tensorkiln::ReferenceResult;
using tensorkiln::ReluOp;
using tensorkiln::Shape;
using tensorkiln::SoftmaxOp;
using tensorkiln::StructuralCanonicalization;
using tensorkiln::StructuralCanonicalizationResult;
using tensorkiln::Tensor;
using tensorkiln::TensorType;
using tensorkiln::TensorView;
using tensorkiln::ValueId;
using tensorkiln::VerifiedGraph;

template <typename T>
[[nodiscard]] T unwrap(tensorkiln::Result<T> result) {
  TK_REQUIRE(result.value_if() != nullptr);
  return std::move(*result.value_if());
}

template <typename T>
const tensorkiln::Diagnostic& require_error(
    const tensorkiln::Result<T>& result, const ErrorCode code) {
  TK_REQUIRE(result.error_if() != nullptr);
  TK_REQUIRE_EQ(result.error_if()->code, code);
  return *result.error_if();
}

[[nodiscard]] TensorType f32(
    const std::initializer_list<std::int64_t> extents) {
  return unwrap(TensorType::create(unwrap(Shape::create(extents))));
}

void require_bits_equal(const std::span<const float> actual,
                        const std::span<const float> expected) {
  TK_REQUIRE_EQ(actual.size(), expected.size());
  for (std::size_t index = 0U; index < actual.size(); ++index) {
    TK_REQUIRE_EQ(std::bit_cast<std::uint32_t>(actual[index]),
                  std::bit_cast<std::uint32_t>(expected[index]));
  }
}

[[nodiscard]] const Tensor& require_reference_output(
    const ReferenceResult& result, const std::string_view name) {
  const Tensor* output = result.output(name);
  TK_REQUIRE(output != nullptr);
  return *output;
}

[[nodiscard]] TensorView require_execution_output(
    const ExecutionSession& session, const std::string_view name) {
  const auto result = session.result();
  TK_REQUIRE(result.has_value());
  const auto output = result->output(name);
  TK_REQUIRE(output.has_value());
  return *output;
}

[[nodiscard]] std::vector<ArenaPlacement> placements_for(
    const GraphArenaLoweringResult& lowered) {
  std::vector<ArenaPlacement> placements;
  placements.reserve(lowered.arena_plan().allocations().size());
  for (const ArenaAllocation& allocation :
       lowered.arena_plan().allocations()) {
    placements.push_back(ArenaPlacement{
        allocation.buffer_ordinal(), allocation.offset_bytes()});
  }
  return placements;
}

TK_TEST("Mul preserves existing public ordinals and appends its new kinds") {
  TK_REQUIRE_EQ(Operation{AddOp{}}.index(), 2U);
  TK_REQUIRE_EQ(Operation{MatMulOp{}}.index(), 3U);
  TK_REQUIRE_EQ(Operation{ReluOp{}}.index(), 4U);
  TK_REQUIRE_EQ(Operation{SoftmaxOp{0U}}.index(), 5U);
  TK_REQUIRE_EQ(Operation{MulOp{}}.index(), 6U);

  TK_REQUIRE_EQ(static_cast<std::uint8_t>(
                    DenseKernelKind::add_contiguous_f32),
                0U);
  TK_REQUIRE_EQ(static_cast<std::uint8_t>(
                    DenseKernelKind::add_broadcast_f32),
                1U);
  TK_REQUIRE_EQ(static_cast<std::uint8_t>(
                    DenseKernelKind::matmul_rank2_f32),
                2U);
  TK_REQUIRE_EQ(static_cast<std::uint8_t>(
                    DenseKernelKind::matmul_batched_f32),
                3U);
  TK_REQUIRE_EQ(static_cast<std::uint8_t>(
                    DenseKernelKind::relu_contiguous_f32),
                4U);
  TK_REQUIRE_EQ(static_cast<std::uint8_t>(
                    DenseKernelKind::softmax_last_axis_f32),
                5U);
  TK_REQUIRE_EQ(static_cast<std::uint8_t>(
                    DenseKernelKind::mul_contiguous_f32),
                6U);
  TK_REQUIRE_EQ(static_cast<std::uint8_t>(
                    DenseKernelKind::mul_broadcast_f32),
                7U);
}

TK_TEST("Mul graph construction is transactional and broadcasts trailing axes") {
  GraphBuilder builder;
  const ValueId left = unwrap(builder.input("left", f32({2, 1, 3})));
  const ValueId incompatible =
      unwrap(builder.input("incompatible", f32({4, 5})));
  const ValueId right = unwrap(builder.input("right", f32({3})));

  require_error(builder.mul(left, incompatible),
                ErrorCode::broadcast_incompatible);
  TK_REQUIRE_EQ(builder.node_count(), 3U);
  const ValueId product = unwrap(builder.mul(left, right));
  TK_REQUIRE_EQ(product.ordinal(), 3U);
  static_cast<void>(unwrap(builder.output("result", product)));
  const VerifiedGraph graph = unwrap(std::move(builder).finish());

  const TensorType* type = graph.type(product);
  TK_REQUIRE(type != nullptr);
  TK_REQUIRE_EQ(type->to_string(), "f32[2,1,3]");
  TK_REQUIRE(std::holds_alternative<MulOp>(graph.nodes()[3].operation()));
  TK_REQUIRE_EQ(graph.nodes()[3].inputs()[0], left);
  TK_REQUIRE_EQ(graph.nodes()[3].inputs()[1], right);
  TK_REQUIRE(graph.dump().find("%3 = mul %0, %2 : f32[2,1,3]") !=
             std::string::npos);

  GraphBuilder foreign_builder;
  const ValueId foreign =
      unwrap(foreign_builder.input("foreign", f32({2, 1, 3})));
  require_error(foreign_builder.mul(foreign, left),
                ErrorCode::value_not_found);
  TK_REQUIRE_EQ(foreign_builder.node_count(), 1U);

  GraphBuilder finished_builder;
  const ValueId finished_value =
      unwrap(finished_builder.input("value", f32({1})));
  static_cast<void>(
      unwrap(finished_builder.output("value", finished_value)));
  static_cast<void>(unwrap(std::move(finished_builder).finish()));
  require_error(finished_builder.mul(finished_value, finished_value),
                ErrorCode::builder_finished);
}

TK_TEST("Mul replay CSE and DCE retain exact operand order") {
  GraphBuilder builder;
  const ValueId left = unwrap(builder.input("left", f32({2, 2})));
  const ValueId right = unwrap(builder.input("right", f32({2, 2})));
  const ValueId forward = unwrap(builder.mul(left, right));
  const ValueId duplicate = unwrap(builder.mul(left, right));
  const ValueId reverse = unwrap(builder.mul(right, left));
  const ValueId combined = unwrap(builder.add(duplicate, reverse));
  const ValueId dead = unwrap(builder.mul(left, right));
  static_cast<void>(forward);
  static_cast<void>(dead);
  static_cast<void>(unwrap(builder.output("result", combined)));
  const VerifiedGraph source = unwrap(std::move(builder).finish());

  const StructuralCanonicalizationResult canonical =
      unwrap(StructuralCanonicalization::run(source));
  TK_REQUIRE_EQ(canonical.stats().source_nodes, 7U);
  TK_REQUIRE_EQ(canonical.stats().result_nodes, 5U);
  TK_REQUIRE_EQ(canonical.stats().common_subexpressions, 2U);
  TK_REQUIRE_EQ(canonical.stats().redundant_relus, 0U);
  TK_REQUIRE(std::holds_alternative<MulOp>(
      canonical.graph().nodes()[2].operation()));
  TK_REQUIRE(std::holds_alternative<MulOp>(
      canonical.graph().nodes()[3].operation()));
  TK_REQUIRE_EQ(canonical.graph().nodes()[2].inputs()[0].ordinal(), 0U);
  TK_REQUIRE_EQ(canonical.graph().nodes()[2].inputs()[1].ordinal(), 1U);
  TK_REQUIRE_EQ(canonical.graph().nodes()[3].inputs()[0].ordinal(), 1U);
  TK_REQUIRE_EQ(canonical.graph().nodes()[3].inputs()[1].ordinal(), 0U);
  const auto* forward_provenance =
      canonical.provenance().for_source(forward);
  TK_REQUIRE(forward_provenance != nullptr);
  TK_REQUIRE(canonical.provenance().for_source(duplicate) ==
             forward_provenance);
  TK_REQUIRE(canonical.provenance().for_source(dead) ==
             forward_provenance);
  TK_REQUIRE(canonical.provenance().for_source(reverse) !=
             forward_provenance);

  const DeadCodeEliminationResult eliminated =
      unwrap(DeadCodeElimination::run(source));
  TK_REQUIRE_EQ(eliminated.stats().source_nodes, 7U);
  TK_REQUIRE_EQ(eliminated.stats().retained_nodes, 5U);
  TK_REQUIRE_EQ(eliminated.stats().removed_nodes, 2U);
  TK_REQUIRE(std::holds_alternative<MulOp>(
      eliminated.graph().nodes()[2].operation()));
  TK_REQUIRE(std::holds_alternative<MulOp>(
      eliminated.graph().nodes()[3].operation()));
  TK_REQUIRE_EQ(eliminated.graph().nodes()[2].inputs()[0].ordinal(), 0U);
  TK_REQUIRE_EQ(eliminated.graph().nodes()[2].inputs()[1].ordinal(), 1U);
  TK_REQUIRE_EQ(eliminated.graph().nodes()[3].inputs()[0].ordinal(), 1U);
  TK_REQUIRE_EQ(eliminated.graph().nodes()[3].inputs()[1].ordinal(), 0U);
  TK_REQUIRE(eliminated.provenance().for_source(forward) == nullptr);
  TK_REQUIRE(eliminated.provenance().for_source(dead) == nullptr);
  TK_REQUIRE(eliminated.provenance().for_source(duplicate) != nullptr);
  TK_REQUIRE(eliminated.provenance().for_source(reverse) != nullptr);

  const std::array<float, 4U> left_data{{-2.0F, 0.5F, 3.0F, -4.0F}};
  const std::array<float, 4U> right_data{{5.0F, -6.0F, 0.25F, -0.5F}};
  const std::array<InputBinding, 2U> bindings{{
      {"left", left_data},
      {"right", right_data},
  }};
  const ReferenceResult source_result =
      unwrap(ReferenceInterpreter::run(source, bindings));
  const ReferenceResult canonical_result =
      unwrap(ReferenceInterpreter::run(canonical.graph(), bindings));
  const ReferenceResult eliminated_result =
      unwrap(ReferenceInterpreter::run(eliminated.graph(), bindings));
  require_bits_equal(require_reference_output(source_result, "result").data(),
                     require_reference_output(canonical_result, "result")
                         .data());
  require_bits_equal(require_reference_output(source_result, "result").data(),
                     require_reference_output(eliminated_result, "result")
                         .data());
}

TK_TEST("Mul reference fixes raw IEEE boundary bits") {
  GraphBuilder builder;
  const ValueId left = unwrap(builder.input("left", f32({6})));
  const ValueId right = unwrap(builder.input("right", f32({6})));
  const ValueId product = unwrap(builder.mul(left, right));
  static_cast<void>(unwrap(builder.output("product", product)));
  const VerifiedGraph graph = unwrap(std::move(builder).finish());

  const std::array<float, 6U> left_data{{
      0.0F,
      -0.0F,
      std::bit_cast<float>(UINT32_C(0x00000001)),
      std::bit_cast<float>(UINT32_C(0x7f800000)),
      std::bit_cast<float>(UINT32_C(0xff800000)),
      std::bit_cast<float>(UINT32_C(0x00800000)),
  }};
  const std::array<float, 6U> right_data{{
      -3.0F, -3.0F, 2.0F, -2.0F, -2.0F, 0.5F,
  }};
  const std::array<std::uint32_t, 6U> expected_bits{{
      UINT32_C(0x80000000),
      UINT32_C(0x00000000),
      UINT32_C(0x00000002),
      UINT32_C(0xff800000),
      UINT32_C(0x7f800000),
      UINT32_C(0x00400000),
  }};
  const std::array<InputBinding, 2U> reference_bindings{{
      {"left", left_data},
      {"right", right_data},
  }};
  const ReferenceResult reference = unwrap(ReferenceInterpreter::run(
      graph, reference_bindings, ReferenceLimits{72U, 18U}));
  const Tensor& reference_output =
      require_reference_output(reference, "product");
  TK_REQUIRE_EQ(reference_output.data().size(), expected_bits.size());
  for (std::size_t index = 0U; index < expected_bits.size(); ++index) {
    TK_REQUIRE_EQ(std::bit_cast<std::uint32_t>(
                      reference_output.data()[index]),
                  expected_bits[index]);
  }

  const ExecutionPlan plan = unwrap(ExecutionPlanCompiler::run(graph));
  TK_REQUIRE_EQ(plan.steps()[0].kernel(),
                DenseKernelKind::mul_contiguous_f32);
  const std::array<ExecutionInputBinding, 2U> execution_bindings{{
      {"left", left_data},
      {"right", right_data},
  }};
  ExecutionSession session = ExecutionSession::create(
      plan, ExecutionSessionOptions{true});
  TK_REQUIRE(session.bind(execution_bindings).has_value());
  TK_REQUIRE_EQ(session.run(), ExecutionRunStatus::success);
  const TensorView actual = require_execution_output(session, "product");
  for (std::size_t index = 0U; index < expected_bits.size(); ++index) {
    TK_REQUIRE_EQ(std::bit_cast<std::uint32_t>(actual.data()[index]),
                  expected_bits[index]);
  }
}

TK_TEST("Mul arena lifetimes agree in forward and reverse reconstruction") {
  GraphBuilder builder;
  const ValueId left = unwrap(builder.input("left", f32({4})));
  const ValueId right = unwrap(builder.input("right", f32({4})));
  const ValueId first = unwrap(builder.mul(left, right));
  const ValueId second = unwrap(builder.mul(first, right));
  static_cast<void>(unwrap(builder.output("result", second)));
  const VerifiedGraph graph = unwrap(std::move(builder).finish());

  const GraphArenaLoweringResult lowered =
      unwrap(GraphArenaLowering::run(graph));
  TK_REQUIRE_EQ(lowered.execution_step_count(), 2U);
  TK_REQUIRE_EQ(lowered.requests().size(), 2U);
  TK_REQUIRE_EQ(lowered.requests()[0],
                (tensorkiln::ArenaBufferRequest{16U, 0U, 2U}));
  TK_REQUIRE_EQ(lowered.requests()[1],
                (tensorkiln::ArenaBufferRequest{16U, 1U, 2U}));
  TK_REQUIRE_EQ(lowered.values_by_buffer_ordinal()[0], first);
  TK_REQUIRE_EQ(lowered.values_by_buffer_ordinal()[1], second);
  TK_REQUIRE(lowered.arena_plan().allocations()[0].offset_bytes() !=
             lowered.arena_plan().allocations()[1].offset_bytes());

  const std::vector<ArenaPlacement> placements = placements_for(lowered);
  const GraphArenaLoweringResult reversed = unwrap(
      GraphArenaPlacementVerifier::verify(graph, placements));
  TK_REQUIRE_EQ(reversed.dump(), lowered.dump());
}

TK_TEST("Mul reference and execution cover scalar contiguous and rank-four broadcast boundaries") {
  GraphBuilder builder;
  const ValueId scalar_left = unwrap(builder.input("scalar_left", f32({})));
  const ValueId scalar_right =
      unwrap(builder.input("scalar_right", f32({})));
  const ValueId scalar = unwrap(builder.mul(scalar_left, scalar_right));

  const ValueId left =
      unwrap(builder.input("left", f32({2, 1, 3, 1})));
  const ValueId right =
      unwrap(builder.input("right", f32({1, 4, 1, 5})));
  const ValueId broadcast = unwrap(builder.mul(left, right));

  const ValueId matrix_left =
      unwrap(builder.input("matrix_left", f32({2, 3})));
  const ValueId matrix_right =
      unwrap(builder.input("matrix_right", f32({2, 3})));
  const ValueId contiguous = unwrap(builder.mul(matrix_left, matrix_right));
  static_cast<void>(unwrap(builder.output("scalar", scalar)));
  static_cast<void>(unwrap(builder.output("broadcast", broadcast)));
  static_cast<void>(unwrap(builder.output("contiguous", contiguous)));
  const VerifiedGraph graph = unwrap(std::move(builder).finish());

  const ExecutionPlan plan = unwrap(ExecutionPlanCompiler::run(graph));
  TK_REQUIRE_EQ(plan.stats().step_count, 3U);
  TK_REQUIRE_EQ(plan.stats().scalar_steps, 127U);
  TK_REQUIRE_EQ(plan.steps()[0].kernel(),
                DenseKernelKind::mul_contiguous_f32);
  TK_REQUIRE_EQ(plan.steps()[1].kernel(),
                DenseKernelKind::mul_broadcast_f32);
  TK_REQUIRE_EQ(plan.steps()[2].kernel(),
                DenseKernelKind::mul_contiguous_f32);
  TK_REQUIRE_EQ(plan.steps()[0].scalar_steps(), 1U);
  TK_REQUIRE_EQ(plan.steps()[1].scalar_steps(), 120U);
  TK_REQUIRE_EQ(plan.steps()[2].scalar_steps(), 6U);
  TK_REQUIRE(plan.dump().find("mul_contiguous_f32(%0,%1) work=1") !=
             std::string::npos);
  TK_REQUIRE(plan.dump().find("mul_broadcast_f32(%3,%4) work=120") !=
             std::string::npos);
  TK_REQUIRE_EQ(plan.arena_plan().allocations().size(), 3U);

  const std::array<float, 1U> scalar_left_data{{-0.0F}};
  const std::array<float, 1U> scalar_right_data{{-2.0F}};
  const std::array<float, 6U> left_data{{1.0F, -2.0F, 3.0F,
                                         -4.0F, 5.0F, -6.0F}};
  std::array<float, 20U> right_data{};
  for (std::size_t index = 0U; index < right_data.size(); ++index) {
    right_data[index] =
        static_cast<float>(static_cast<std::int32_t>(index) - 9) * 0.25F;
  }
  const std::array<float, 6U> matrix_left_data{{-2.0F, -1.0F, -0.0F,
                                                0.0F, 1.5F, 4.0F}};
  const std::array<float, 6U> matrix_right_data{{-3.0F, 2.0F, -5.0F,
                                                 -7.0F, -2.0F, 0.5F}};
  const std::array<InputBinding, 6U> reference_bindings{{
      {"scalar_left", scalar_left_data},
      {"scalar_right", scalar_right_data},
      {"left", left_data},
      {"right", right_data},
      {"matrix_left", matrix_left_data},
      {"matrix_right", matrix_right_data},
  }};

  const ReferenceLimits exact_limits{668U, 167U};
  const ReferenceResult reference = unwrap(
      ReferenceInterpreter::run(graph, reference_bindings, exact_limits));
  require_error(ReferenceInterpreter::run(
                    graph, reference_bindings, ReferenceLimits{667U, 167U}),
                ErrorCode::reference_materialization_limit_exceeded);
  require_error(ReferenceInterpreter::run(
                    graph, reference_bindings, ReferenceLimits{668U, 166U}),
                ErrorCode::reference_scalar_step_limit_exceeded);
  TK_REQUIRE_EQ(reference.materialized_bytes(), 668U);
  TK_REQUIRE_EQ(reference.scalar_steps(), 167U);
  TK_REQUIRE_EQ(std::bit_cast<std::uint32_t>(
                    require_reference_output(reference, "scalar").data()[0]),
                0U);

  std::vector<float> expected_broadcast;
  expected_broadcast.reserve(120U);
  for (std::size_t batch = 0U; batch < 2U; ++batch) {
    for (std::size_t group = 0U; group < 4U; ++group) {
      for (std::size_t channel = 0U; channel < 3U; ++channel) {
        for (std::size_t column = 0U; column < 5U; ++column) {
          expected_broadcast.push_back(
              left_data[batch * 3U + channel] *
              right_data[group * 5U + column]);
        }
      }
    }
  }
  require_bits_equal(
      require_reference_output(reference, "broadcast").data(),
      expected_broadcast);
  const std::array<float, 6U> expected_contiguous{{
      6.0F, -2.0F, 0.0F, -0.0F, -3.0F, 2.0F,
  }};
  require_bits_equal(
      require_reference_output(reference, "contiguous").data(),
      expected_contiguous);

  const std::array<ExecutionInputBinding, 6U> execution_bindings{{
      {"scalar_left", scalar_left_data},
      {"scalar_right", scalar_right_data},
      {"left", left_data},
      {"right", right_data},
      {"matrix_left", matrix_left_data},
      {"matrix_right", matrix_right_data},
  }};
  ExecutionSession session = ExecutionSession::create(
      plan, ExecutionSessionOptions{true});
  TK_REQUIRE(session.bind(execution_bindings).has_value());
  TK_REQUIRE_EQ(session.run(), ExecutionRunStatus::success);
  for (const std::string_view name :
       {"scalar", "broadcast", "contiguous"}) {
    require_bits_equal(require_execution_output(session, name).data(),
                       require_reference_output(reference, name).data());
  }
  TK_REQUIRE_EQ(session.run(), ExecutionRunStatus::success);
  for (const std::string_view name :
       {"scalar", "broadcast", "contiguous"}) {
    require_bits_equal(require_execution_output(session, name).data(),
                       require_reference_output(reference, name).data());
  }
}

TK_TEST("Mul reverse verifier independently rejects the wrong kernel class") {
  GraphBuilder builder;
  const ValueId left = unwrap(builder.input("left", f32({2, 3})));
  const ValueId right = unwrap(builder.input("right", f32({3})));
  const ValueId product = unwrap(builder.mul(left, right));
  static_cast<void>(unwrap(builder.output("product", product)));
  const VerifiedGraph graph = unwrap(std::move(builder).finish());
  const GraphArenaLoweringResult lowered =
      unwrap(GraphArenaLowering::run(graph));
  TK_REQUIRE_EQ(lowered.execution_step_count(), 1U);
  TK_REQUIRE_EQ(lowered.values_by_buffer_ordinal().size(), 1U);
  TK_REQUIRE_EQ(lowered.values_by_buffer_ordinal()[0], product);
  const std::vector<ArenaPlacement> placements = placements_for(lowered);

  const std::array<ExecutionStepSpec, 1U> correct{{
      {2U, DenseKernelKind::mul_broadcast_f32},
  }};
  const ExecutionPlan verified = unwrap(ExecutionPlanVerifier::verify(
      graph, ExecutionPlanCandidate{correct, placements}));
  TK_REQUIRE_EQ(verified.steps()[0].kernel(),
                DenseKernelKind::mul_broadcast_f32);
  TK_REQUIRE_EQ(verified.steps()[0].scalar_steps(), 6U);

  tensorkiln::ExecutionPlanLimits exact_limits;
  exact_limits.max_values = 3U;
  exact_limits.max_steps = 1U;
  exact_limits.max_outputs = 1U;
  exact_limits.max_owned_constant_bytes = 0U;
  exact_limits.max_scalar_steps = 6U;
  TK_REQUIRE(ExecutionPlanVerifier::verify(
                 graph, ExecutionPlanCandidate{correct, placements},
                 exact_limits)
                 .has_value());
  exact_limits.max_scalar_steps = 5U;
  require_error(ExecutionPlanVerifier::verify(
                    graph, ExecutionPlanCandidate{correct, placements},
                    exact_limits),
                ErrorCode::plan_scalar_step_limit_exceeded);
  require_error(ExecutionPlanCompiler::run(graph, exact_limits),
                ErrorCode::plan_scalar_step_limit_exceeded);

  const std::array<ExecutionStepSpec, 1U> wrong_class{{
      {2U, DenseKernelKind::mul_contiguous_f32},
  }};
  const auto rejected = ExecutionPlanVerifier::verify(
      graph, ExecutionPlanCandidate{wrong_class, placements});
  const auto& diagnostic =
      require_error(rejected, ErrorCode::plan_kernel_incompatible);
  TK_REQUIRE(diagnostic.message.find("expected mul_broadcast_f32") !=
             std::string::npos);

  const std::array<ExecutionStepSpec, 1U> wrong_operation{{
      {2U, DenseKernelKind::add_broadcast_f32},
  }};
  require_error(ExecutionPlanVerifier::verify(
                    graph, ExecutionPlanCandidate{wrong_operation, placements}),
                ErrorCode::plan_kernel_incompatible);
}

}  // namespace
