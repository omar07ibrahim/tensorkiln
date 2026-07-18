#include "test.hpp"

#include <array>
#include <cstdint>
#include <initializer_list>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "tensorkiln/execution_plan.hpp"

namespace {

using tensorkiln::DenseKernelKind;
using tensorkiln::ExecutionPlan;
using tensorkiln::ExecutionPlanCompiler;
using tensorkiln::ExecutionPlanLimits;
using tensorkiln::GraphBuilder;
using tensorkiln::PlanStorageKind;
using tensorkiln::Shape;
using tensorkiln::TensorType;
using tensorkiln::ValueId;
using tensorkiln::VerifiedGraph;

template <typename T>
[[nodiscard]] T unwrap(tensorkiln::Result<T> result) {
  TK_REQUIRE(result.value_if() != nullptr);
  return std::move(*result.value_if());
}

[[nodiscard]] TensorType f32(
    const std::initializer_list<std::int64_t> extents) {
  return unwrap(TensorType::create(unwrap(Shape::create(extents))));
}

struct MixedFixture final {
  VerifiedGraph graph;
  ValueId input;
  ValueId weight;
  ValueId product;
  ValueId bias;
  ValueId sum;
  ValueId result;
};

[[nodiscard]] MixedFixture make_mixed_fixture() {
  GraphBuilder builder;
  const ValueId input = unwrap(builder.input("x", f32({2, 3})));
  const std::array<float, 12U> weight_data{{
      1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F,
      7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F,
  }};
  const ValueId weight = unwrap(
      builder.constant("weight", f32({3, 4}), weight_data));
  const ValueId product = unwrap(builder.matmul(input, weight));
  const std::array<float, 4U> bias_data{{1.0F, 2.0F, 3.0F, 4.0F}};
  const ValueId bias =
      unwrap(builder.constant("bias", f32({4}), bias_data));
  const ValueId sum = unwrap(builder.add(product, bias));
  const ValueId result = unwrap(builder.relu(sum));
  static_cast<void>(unwrap(builder.output("result", result)));
  static_cast<void>(unwrap(builder.output("product", product)));
  return MixedFixture{
      unwrap(std::move(builder).finish()),
      input,
      weight,
      product,
      bias,
      sum,
      result,
  };
}

TK_TEST("Execution plan compiler seals dense layouts storage and kernels") {
  const MixedFixture fixture = make_mixed_fixture();
  const std::string source_dump = fixture.graph.dump();
  const ExecutionPlan plan =
      unwrap(ExecutionPlanCompiler::run(fixture.graph));

  TK_REQUIRE_EQ(fixture.graph.dump(), source_dump);
  TK_REQUIRE_EQ(plan.stats().value_count, 6U);
  TK_REQUIRE_EQ(plan.stats().input_count, 1U);
  TK_REQUIRE_EQ(plan.stats().constant_count, 2U);
  TK_REQUIRE_EQ(plan.stats().step_count, 3U);
  TK_REQUIRE_EQ(plan.stats().output_count, 2U);
  TK_REQUIRE_EQ(plan.stats().owned_constant_bytes, 64U);
  TK_REQUIRE_EQ(plan.stats().scalar_steps, 40U);
  TK_REQUIRE_EQ(plan.stats().workspace_bytes,
                plan.arena_plan().workspace_bytes());

  TK_REQUIRE_EQ(plan.values().size(), 6U);
  const auto input_strides = plan.values()[0].layout().strides_elements();
  TK_REQUIRE_EQ(input_strides.size(), 2U);
  TK_REQUIRE_EQ(input_strides[0], 3U);
  TK_REQUIRE_EQ(input_strides[1], 1U);
  TK_REQUIRE_EQ(plan.values()[0].layout().elements(), 6U);
  TK_REQUIRE_EQ(plan.values()[0].storage().kind(), PlanStorageKind::input);
  TK_REQUIRE_EQ(plan.values()[0].storage().ordinal(), 0U);
  TK_REQUIRE_EQ(plan.values()[1].storage().kind(),
                PlanStorageKind::constant);
  TK_REQUIRE_EQ(plan.values()[1].storage().ordinal(), 0U);
  TK_REQUIRE_EQ(plan.values()[3].storage().kind(),
                PlanStorageKind::constant);
  TK_REQUIRE_EQ(plan.values()[3].storage().ordinal(), 1U);
  TK_REQUIRE_EQ(plan.values()[2].storage().kind(), PlanStorageKind::arena);
  TK_REQUIRE_EQ(plan.values()[2].storage().ordinal(), 0U);
  TK_REQUIRE_EQ(plan.values()[4].storage().ordinal(), 1U);
  TK_REQUIRE_EQ(plan.values()[5].storage().ordinal(), 2U);

  TK_REQUIRE_EQ(plan.steps().size(), 3U);
  TK_REQUIRE_EQ(plan.steps()[0].source_node().ordinal(), 2U);
  TK_REQUIRE_EQ(plan.steps()[0].kernel(),
                DenseKernelKind::matmul_rank2_f32);
  TK_REQUIRE_EQ(plan.steps()[0].scalar_steps(), 24U);
  TK_REQUIRE_EQ(plan.steps()[1].source_node().ordinal(), 4U);
  TK_REQUIRE_EQ(plan.steps()[1].kernel(),
                DenseKernelKind::add_broadcast_f32);
  TK_REQUIRE_EQ(plan.steps()[1].scalar_steps(), 8U);
  TK_REQUIRE_EQ(plan.steps()[2].kernel(),
                DenseKernelKind::relu_contiguous_f32);
  TK_REQUIRE_EQ(plan.steps()[2].scalar_steps(), 8U);
  TK_REQUIRE_EQ(plan.steps()[0].operands().size(), 2U);
  TK_REQUIRE_EQ(plan.steps()[0].operands()[0], fixture.input);
  TK_REQUIRE_EQ(plan.steps()[0].operands()[1], fixture.weight);
  TK_REQUIRE_EQ(plan.steps()[0].output(), fixture.product);
  TK_REQUIRE_EQ(plan.steps()[1].operands()[1], fixture.bias);
  TK_REQUIRE_EQ(plan.steps()[2].output(), fixture.result);
}

TK_TEST("Execution plan dump is stable across calls and compilations") {
  const MixedFixture first_fixture = make_mixed_fixture();
  const ExecutionPlan first =
      unwrap(ExecutionPlanCompiler::run(first_fixture.graph));
  const std::string first_dump = first.dump();
  TK_REQUIRE_EQ(first.dump(), first_dump);

  const MixedFixture second_fixture = make_mixed_fixture();
  const ExecutionPlan second =
      unwrap(ExecutionPlanCompiler::run(second_fixture.graph));
  TK_REQUIRE_EQ(second.dump(), first_dump);
  TK_REQUIRE(first_dump.find("matmul_rank2_f32(%0,%1) work=24") !=
             std::string::npos);
  TK_REQUIRE(first_dump.find("add_broadcast_f32(%2,%3) work=8") !=
             std::string::npos);
  TK_REQUIRE(first_dump.find("product -> %2") != std::string::npos);
}

struct OwnedPlan final {
  ExecutionPlan plan;
  ValueId input;
};

[[nodiscard]] OwnedPlan make_owned_plan() {
  GraphBuilder builder;
  const ValueId input = unwrap(builder.input("x", f32({2})));
  const ValueId result = unwrap(builder.relu(input));
  static_cast<void>(unwrap(builder.output("result", result)));
  const VerifiedGraph graph = unwrap(std::move(builder).finish());
  return OwnedPlan{unwrap(ExecutionPlanCompiler::run(graph)), input};
}

TK_TEST("Execution plan owns its graph and survives moves") {
  OwnedPlan owned = make_owned_plan();
  const std::string expected_dump = owned.plan.dump();
  TK_REQUIRE(owned.plan.value(owned.input) != nullptr);

  ExecutionPlan moved = std::move(owned.plan);
  TK_REQUIRE_EQ(moved.dump(), expected_dump);
  TK_REQUIRE(moved.value(owned.input) != nullptr);
  TK_REQUIRE(moved.graph().type(owned.input) != nullptr);

  GraphBuilder foreign_builder;
  const ValueId foreign =
      unwrap(foreign_builder.input("foreign", f32({2})));
  static_cast<void>(unwrap(foreign_builder.output("foreign", foreign)));
  const VerifiedGraph foreign_graph =
      unwrap(std::move(foreign_builder).finish());
  TK_REQUIRE(foreign_graph.type(foreign) != nullptr);
  TK_REQUIRE(moved.value(foreign) == nullptr);
}

TK_TEST("Execution plan accepts external-only graphs at exact zero work limits") {
  GraphBuilder builder;
  const ValueId input = unwrap(builder.input("x", f32({2})));
  const std::array<float, 2U> data{{3.0F, 4.0F}};
  const ValueId constant =
      unwrap(builder.constant("c", f32({2}), data));
  static_cast<void>(unwrap(builder.output("input_result", input)));
  static_cast<void>(unwrap(builder.output("constant_result", constant)));
  const VerifiedGraph graph = unwrap(std::move(builder).finish());

  ExecutionPlanLimits limits;
  limits.max_values = 2U;
  limits.max_steps = 0U;
  limits.max_outputs = 2U;
  limits.max_owned_constant_bytes = 8U;
  limits.max_scalar_steps = 0U;
  limits.arena_limits = tensorkiln::ArenaLimits{0U, 0U};
  const ExecutionPlan plan =
      unwrap(ExecutionPlanCompiler::run(graph, limits));

  TK_REQUIRE_EQ(plan.stats(),
                (tensorkiln::ExecutionPlanStats{
                    2U, 1U, 1U, 0U, 2U, 8U, 0U, 0U}));
  TK_REQUIRE(plan.steps().empty());
  TK_REQUIRE(plan.arena_plan().allocations().empty());
  TK_REQUIRE_EQ(plan.values()[0].storage().kind(), PlanStorageKind::input);
  TK_REQUIRE_EQ(plan.values()[1].storage().kind(),
                PlanStorageKind::constant);
  TK_REQUIRE(plan.dump().find("  steps {\n  }\n") != std::string::npos);
  TK_REQUIRE(plan.dump().find("  arena {\n  }\n") != std::string::npos);
}

TK_TEST("Execution plan classifies contiguous add and batched matmul") {
  GraphBuilder builder;
  const ValueId left = unwrap(builder.input("left", f32({2, 2, 3})));
  const ValueId right = unwrap(builder.input("right", f32({2, 3, 4})));
  const ValueId product = unwrap(builder.matmul(left, right));
  const ValueId other = unwrap(builder.input("other", f32({2, 2, 4})));
  const ValueId sum = unwrap(builder.add(product, other));
  static_cast<void>(unwrap(builder.output("sum", sum)));
  const VerifiedGraph graph = unwrap(std::move(builder).finish());
  const ExecutionPlan plan = unwrap(ExecutionPlanCompiler::run(graph));

  TK_REQUIRE_EQ(plan.steps().size(), 2U);
  TK_REQUIRE_EQ(plan.steps()[0].kernel(),
                DenseKernelKind::matmul_batched_f32);
  TK_REQUIRE_EQ(plan.steps()[1].kernel(),
                DenseKernelKind::add_contiguous_f32);
}

TK_TEST("Execution plan represents a scalar with an empty dense stride list") {
  GraphBuilder builder;
  const TensorType scalar =
      unwrap(TensorType::create(Shape::scalar()));
  const ValueId input = unwrap(builder.input("scalar", scalar));
  static_cast<void>(unwrap(builder.output("scalar", input)));
  const VerifiedGraph graph = unwrap(std::move(builder).finish());
  const ExecutionPlan plan = unwrap(ExecutionPlanCompiler::run(graph));

  TK_REQUIRE_EQ(plan.values().size(), 1U);
  TK_REQUIRE_EQ(plan.values()[0].layout().rank(), 0U);
  TK_REQUIRE(plan.values()[0].layout().strides_elements().empty());
  TK_REQUIRE_EQ(plan.values()[0].layout().elements(), 1U);
}

}  // namespace
