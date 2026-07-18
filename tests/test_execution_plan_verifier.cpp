#include "test.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include "tensorkiln/execution_plan.hpp"

namespace {

using tensorkiln::ArenaPlacement;
using tensorkiln::DenseKernelKind;
using tensorkiln::Diagnostic;
using tensorkiln::ErrorCode;
using tensorkiln::ExecutionPlan;
using tensorkiln::ExecutionPlanCandidate;
using tensorkiln::ExecutionPlanCompiler;
using tensorkiln::ExecutionPlanLimits;
using tensorkiln::ExecutionPlanVerifier;
using tensorkiln::ExecutionStepSpec;
using tensorkiln::GraphArenaLowering;
using tensorkiln::GraphBuilder;
using tensorkiln::Shape;
using tensorkiln::TensorType;
using tensorkiln::ValueId;
using tensorkiln::VerifiedGraph;

template <typename T>
[[nodiscard]] T unwrap(tensorkiln::Result<T> result) {
  TK_REQUIRE(result.value_if() != nullptr);
  return std::move(*result.value_if());
}

const Diagnostic& require_error(
    const tensorkiln::Result<ExecutionPlan>& result,
    const ErrorCode code) {
  TK_REQUIRE(result.error_if() != nullptr);
  TK_REQUIRE_EQ(result.error_if()->code, code);
  return *result.error_if();
}

[[nodiscard]] TensorType f32(
    const std::initializer_list<std::int64_t> extents) {
  return unwrap(TensorType::create(unwrap(Shape::create(extents))));
}

struct CandidateFixture final {
  VerifiedGraph graph;
  std::vector<ExecutionStepSpec> specs;
  std::vector<ArenaPlacement> placements;
};

[[nodiscard]] CandidateFixture make_candidate_fixture() {
  GraphBuilder builder;
  const ValueId input = unwrap(builder.input("x", f32({2, 3})));
  const std::array<float, 3U> bias_data{{1.0F, 2.0F, 3.0F}};
  const ValueId bias =
      unwrap(builder.constant("bias", f32({3}), bias_data));
  const ValueId sum = unwrap(builder.add(input, bias));
  const ValueId result = unwrap(builder.relu(sum));
  static_cast<void>(unwrap(builder.output("result", result)));
  VerifiedGraph graph = unwrap(std::move(builder).finish());

  const auto lowered = unwrap(GraphArenaLowering::run(graph));
  std::vector<ArenaPlacement> placements;
  placements.reserve(lowered.arena_plan().allocations().size());
  for (const tensorkiln::ArenaAllocation& allocation :
       lowered.arena_plan().allocations()) {
    placements.push_back(
        ArenaPlacement{allocation.buffer_ordinal(),
                       allocation.offset_bytes()});
  }
  std::vector<ExecutionStepSpec> specs{
      {2U, DenseKernelKind::add_broadcast_f32},
      {3U, DenseKernelKind::relu_contiguous_f32},
  };
  return CandidateFixture{
      std::move(graph),
      std::move(specs),
      std::move(placements),
  };
}

[[nodiscard]] ExecutionPlanCandidate candidate_for(
    const std::vector<ExecutionStepSpec>& specs,
    const std::vector<ArenaPlacement>& placements) {
  return ExecutionPlanCandidate{
      std::span<const ExecutionStepSpec>{specs},
      std::span<const ArenaPlacement>{placements},
  };
}

TK_TEST("Execution plan verifier accepts an independently supplied candidate") {
  const CandidateFixture fixture = make_candidate_fixture();
  const ExecutionPlan verified = unwrap(ExecutionPlanVerifier::verify(
      fixture.graph, candidate_for(fixture.specs, fixture.placements)));
  const ExecutionPlan compiled =
      unwrap(ExecutionPlanCompiler::run(fixture.graph));
  TK_REQUIRE_EQ(verified.dump(), compiled.dump());
}

TK_TEST("Execution plan verifier rejects missing and extra step specs") {
  const CandidateFixture fixture = make_candidate_fixture();
  std::vector<ExecutionStepSpec> missing = fixture.specs;
  missing.pop_back();
  require_error(ExecutionPlanVerifier::verify(
                    fixture.graph,
                    candidate_for(missing, fixture.placements)),
                ErrorCode::plan_step_count_mismatch);

  std::vector<ExecutionStepSpec> extra = fixture.specs;
  extra.push_back(fixture.specs.back());
  require_error(ExecutionPlanVerifier::verify(
                    fixture.graph,
                    candidate_for(extra, fixture.placements)),
                ErrorCode::plan_step_count_mismatch);
}

TK_TEST("Execution plan verifier rejects reordered or foreign step sources") {
  const CandidateFixture fixture = make_candidate_fixture();
  std::vector<ExecutionStepSpec> reordered = fixture.specs;
  std::swap(reordered[0], reordered[1]);
  require_error(ExecutionPlanVerifier::verify(
                    fixture.graph,
                    candidate_for(reordered, fixture.placements)),
                ErrorCode::plan_step_source_mismatch);

  std::vector<ExecutionStepSpec> foreign = fixture.specs;
  foreign[0].source_node_ordinal = 99U;
  require_error(ExecutionPlanVerifier::verify(
                    fixture.graph,
                    candidate_for(foreign, fixture.placements)),
                ErrorCode::plan_step_source_mismatch);
}

TK_TEST("Execution plan verifier rejects invalid and incompatible kernels") {
  const CandidateFixture fixture = make_candidate_fixture();
  std::vector<ExecutionStepSpec> invalid = fixture.specs;
  invalid[0].kernel = static_cast<DenseKernelKind>(255U);
  require_error(ExecutionPlanVerifier::verify(
                    fixture.graph,
                    candidate_for(invalid, fixture.placements)),
                ErrorCode::plan_kernel_invalid);

  std::vector<ExecutionStepSpec> incompatible = fixture.specs;
  incompatible[0].kernel = DenseKernelKind::add_contiguous_f32;
  require_error(ExecutionPlanVerifier::verify(
                    fixture.graph,
                    candidate_for(incompatible, fixture.placements)),
                ErrorCode::plan_kernel_incompatible);

  incompatible = fixture.specs;
  incompatible[1].kernel = DenseKernelKind::matmul_rank2_f32;
  require_error(ExecutionPlanVerifier::verify(
                    fixture.graph,
                    candidate_for(incompatible, fixture.placements)),
                ErrorCode::plan_kernel_incompatible);
}

TK_TEST("Execution plan verifier delegates placement proof to arena verifier") {
  const CandidateFixture fixture = make_candidate_fixture();
  std::vector<ArenaPlacement> missing = fixture.placements;
  missing.pop_back();
  require_error(ExecutionPlanVerifier::verify(
                    fixture.graph,
                    candidate_for(fixture.specs, missing)),
                ErrorCode::arena_placement_count_mismatch);

  std::vector<ArenaPlacement> unaligned = fixture.placements;
  ++unaligned[0].offset_bytes;
  require_error(ExecutionPlanVerifier::verify(
                    fixture.graph,
                    candidate_for(fixture.specs, unaligned)),
                ErrorCode::arena_alignment_invalid);

  std::vector<ArenaPlacement> overlapping = fixture.placements;
  overlapping[0].offset_bytes = 0U;
  overlapping[1].offset_bytes = 0U;
  require_error(ExecutionPlanVerifier::verify(
                    fixture.graph,
                    candidate_for(fixture.specs, overlapping)),
                ErrorCode::arena_live_overlap);
}

TK_TEST("Execution plan verifier enforces exact and one-below policy limits") {
  const CandidateFixture fixture = make_candidate_fixture();
  ExecutionPlanLimits exact;
  exact.max_values = 4U;
  exact.max_steps = 2U;
  exact.max_outputs = 1U;
  exact.max_owned_constant_bytes = 12U;
  exact.max_scalar_steps = 12U;
  TK_REQUIRE(ExecutionPlanVerifier::verify(
                 fixture.graph,
                 candidate_for(fixture.specs, fixture.placements), exact)
                 .has_value());

  ExecutionPlanLimits limited = exact;
  limited.max_values = 3U;
  require_error(ExecutionPlanVerifier::verify(
                    fixture.graph,
                    candidate_for(fixture.specs, fixture.placements), limited),
                ErrorCode::plan_value_limit_exceeded);

  limited = exact;
  limited.max_steps = 1U;
  require_error(ExecutionPlanVerifier::verify(
                    fixture.graph,
                    candidate_for(fixture.specs, fixture.placements), limited),
                ErrorCode::plan_step_limit_exceeded);

  limited = exact;
  limited.max_outputs = 0U;
  require_error(ExecutionPlanVerifier::verify(
                    fixture.graph,
                    candidate_for(fixture.specs, fixture.placements), limited),
                ErrorCode::plan_output_limit_exceeded);

  limited = exact;
  limited.max_owned_constant_bytes = 11U;
  require_error(ExecutionPlanVerifier::verify(
                    fixture.graph,
                    candidate_for(fixture.specs, fixture.placements), limited),
                ErrorCode::plan_constant_byte_limit_exceeded);

  limited = exact;
  limited.max_scalar_steps = 11U;
  require_error(ExecutionPlanVerifier::verify(
                    fixture.graph,
                    candidate_for(fixture.specs, fixture.placements), limited),
                ErrorCode::plan_scalar_step_limit_exceeded);

  limited = exact;
  limited.max_steps = 1U;
  require_error(ExecutionPlanCompiler::run(fixture.graph, limited),
                ErrorCode::plan_step_limit_exceeded);
}

TK_TEST("Oversized plan work fails before materialization") {
  constexpr std::uint64_t extent = UINT64_C(1) << 22U;
  constexpr std::uint64_t max_elements = UINT64_C(1) << 44U;
  constexpr std::uint64_t max_bytes = UINT64_C(1) << 46U;
  const tensorkiln::ShapeLimits shape_limits{max_elements};
  const tensorkiln::TensorLimits tensor_limits{max_bytes};
  const Shape huge_shape = unwrap(Shape::create(
      {static_cast<std::int64_t>(extent),
       static_cast<std::int64_t>(extent)},
      shape_limits));
  auto huge_type_result = TensorType::create(
      huge_shape, tensorkiln::ElementType::f32, tensor_limits);
  if (max_bytes > static_cast<std::uint64_t>(
                      std::numeric_limits<std::size_t>::max())) {
    TK_REQUIRE(huge_type_result.error_if() != nullptr);
    TK_REQUIRE_EQ(huge_type_result.error_if()->code,
                  ErrorCode::byte_count_overflow);
    return;
  }
  const TensorType huge_type = unwrap(std::move(huge_type_result));

  tensorkiln::GraphLimits graph_limits;
  graph_limits.shape_limits = shape_limits;
  graph_limits.tensor_limits = tensor_limits;
  GraphBuilder builder(graph_limits);
  const ValueId left = unwrap(builder.input("left", huge_type));
  const ValueId right = unwrap(builder.input("right", huge_type));
  const ValueId product = unwrap(builder.matmul(left, right));
  static_cast<void>(unwrap(builder.output("product", product)));
  const VerifiedGraph graph = unwrap(std::move(builder).finish());

  require_error(ExecutionPlanCompiler::run(graph),
                ErrorCode::plan_work_overflow);
  const std::array<ExecutionStepSpec, 0U> specs{};
  const std::array<ArenaPlacement, 0U> placements{};
  require_error(
      ExecutionPlanVerifier::verify(
          graph,
          ExecutionPlanCandidate{
              std::span<const ExecutionStepSpec>{specs},
              std::span<const ArenaPlacement>{placements},
          }),
      ErrorCode::plan_work_overflow);
}

TK_TEST("Execution plan verifier accepts an external-only empty candidate") {
  GraphBuilder builder;
  const ValueId input = unwrap(builder.input("x", f32({1})));
  static_cast<void>(unwrap(builder.output("result", input)));
  const VerifiedGraph graph = unwrap(std::move(builder).finish());
  const std::array<ExecutionStepSpec, 0U> specs{};
  const std::array<ArenaPlacement, 0U> placements{};
  ExecutionPlanLimits limits;
  limits.max_values = 1U;
  limits.max_steps = 0U;
  limits.max_outputs = 1U;
  limits.max_owned_constant_bytes = 0U;
  limits.max_scalar_steps = 0U;
  limits.arena_limits = tensorkiln::ArenaLimits{0U, 0U};
  const ExecutionPlan plan = unwrap(ExecutionPlanVerifier::verify(
      graph,
      ExecutionPlanCandidate{
          std::span<const ExecutionStepSpec>{specs},
          std::span<const ArenaPlacement>{placements},
      },
      limits));
  TK_REQUIRE(plan.steps().empty());
  TK_REQUIRE_EQ(plan.stats().workspace_bytes, 0U);
}

TK_TEST("Execution plan diagnostics expose stable typed names") {
  const std::array<std::pair<ErrorCode, std::string_view>, 10U> cases{{
      {ErrorCode::plan_value_limit_exceeded,
       "plan_value_limit_exceeded"},
      {ErrorCode::plan_step_limit_exceeded,
       "plan_step_limit_exceeded"},
      {ErrorCode::plan_output_limit_exceeded,
       "plan_output_limit_exceeded"},
      {ErrorCode::plan_constant_byte_limit_exceeded,
       "plan_constant_byte_limit_exceeded"},
      {ErrorCode::plan_work_overflow, "plan_work_overflow"},
      {ErrorCode::plan_scalar_step_limit_exceeded,
       "plan_scalar_step_limit_exceeded"},
      {ErrorCode::plan_step_count_mismatch,
       "plan_step_count_mismatch"},
      {ErrorCode::plan_step_source_mismatch,
       "plan_step_source_mismatch"},
      {ErrorCode::plan_kernel_invalid, "plan_kernel_invalid"},
      {ErrorCode::plan_kernel_incompatible,
       "plan_kernel_incompatible"},
  }};
  for (const auto& [code, expected] : cases) {
    TK_REQUIRE_EQ(tensorkiln::error_code_name(code), expected);
  }
}

}  // namespace
