#include "test.hpp"

#include <array>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "tensorkiln/dead_code_elimination.hpp"
#include "tensorkiln/graph_arena.hpp"
#include "../src/graph_arena_lowering_internal.hpp"

namespace {

using tensorkiln::ArenaLimits;
using tensorkiln::ArenaAllocation;
using tensorkiln::ArenaPlacement;
using tensorkiln::ArenaPlanner;
using tensorkiln::Diagnostic;
using tensorkiln::ErrorCode;
using tensorkiln::GraphArenaLowering;
using tensorkiln::GraphArenaLoweringResult;
using tensorkiln::GraphArenaPlacementVerifier;
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

[[nodiscard]] const Diagnostic& require_error(
    const tensorkiln::Result<GraphArenaLoweringResult>& result,
    const ErrorCode code) {
  TK_REQUIRE(result.error_if() != nullptr);
  TK_REQUIRE_EQ(result.error_if()->code, code);
  return *result.error_if();
}

[[nodiscard]] std::uint32_t require_buffer_ordinal(
    const GraphArenaLoweringResult& result, const ValueId value) {
  const std::optional<std::uint32_t> ordinal =
      result.buffer_ordinal(value);
  TK_REQUIRE(ordinal.has_value());
  return *ordinal;
}

[[nodiscard]] const ValueId& require_value_at(
    const GraphArenaLoweringResult& result,
    const std::uint32_t buffer_ordinal) {
  const ValueId* value = result.value_at(buffer_ordinal);
  TK_REQUIRE(value != nullptr);
  return *value;
}

[[nodiscard]] const ArenaAllocation& require_allocation(
    const GraphArenaLoweringResult& result, const ValueId value) {
  const ArenaAllocation* allocation = result.allocation_for(value);
  TK_REQUIRE(allocation != nullptr);
  return *allocation;
}

void require_plan_parity(const VerifiedGraph& graph,
                         const GraphArenaLoweringResult& lowered,
                         const ArenaLimits limits = ArenaLimits{}) {
  const tensorkiln::ArenaPlan replanned =
      unwrap(ArenaPlanner::run(lowered.requests(), limits));
  TK_REQUIRE_EQ(replanned.dump(), lowered.arena_plan().dump());

  std::vector<ArenaPlacement> placements;
  placements.reserve(lowered.arena_plan().allocations().size());
  for (const ArenaAllocation& allocation :
       lowered.arena_plan().allocations()) {
    placements.push_back(ArenaPlacement{
        allocation.buffer_ordinal(),
        allocation.offset_bytes(),
    });
  }
  const GraphArenaLoweringResult reverified = unwrap(
      GraphArenaPlacementVerifier::verify(graph, placements, limits));
  TK_REQUIRE_EQ(reverified.dump(), lowered.dump());
}

[[nodiscard]] TensorType f32(
    const std::initializer_list<std::int64_t> extents) {
  return unwrap(TensorType::create(unwrap(Shape::create(extents))));
}

struct ChainFixture final {
  VerifiedGraph graph;
  ValueId input;
  ValueId first;
  ValueId second;
  ValueId third;
};

[[nodiscard]] ChainFixture make_chain() {
  GraphBuilder builder;
  const ValueId input = unwrap(builder.input("x", f32({16})));
  const ValueId first = unwrap(builder.relu(input));
  const ValueId second = unwrap(builder.relu(first));
  const ValueId third = unwrap(builder.relu(second));
  static_cast<void>(unwrap(builder.output("result", third)));
  static_cast<void>(unwrap(builder.output("result_alias", third)));
  return ChainFixture{
      unwrap(std::move(builder).finish()),
      input,
      first,
      second,
      third,
  };
}

[[nodiscard]] std::array<ArenaPlacement, 3U> chain_placements() {
  return {{
      {2U, 0U},
      {1U, 64U},
      {0U, 0U},
  }};
}

struct MixedFixture final {
  VerifiedGraph graph;
  ValueId input;
  ValueId weight;
  ValueId product;
  ValueId early;
  ValueId bias;
  ValueId dead;
  ValueId result;
};

[[nodiscard]] MixedFixture make_mixed_graph() {
  GraphBuilder builder;
  const ValueId input = unwrap(builder.input("x", f32({1, 1})));
  const std::array<float, 1U> weight_data{{2.0F}};
  const ValueId weight =
      unwrap(builder.constant("weight", f32({1, 1}), weight_data));
  const ValueId product = unwrap(builder.matmul(input, weight));
  const ValueId early = unwrap(builder.relu(product));
  const std::array<float, 1U> bias_data{{1.0F}};
  const ValueId bias =
      unwrap(builder.constant("bias", f32({1, 1}), bias_data));
  const ValueId dead = unwrap(builder.add(early, bias));
  const ValueId result = unwrap(builder.add(product, bias));
  static_cast<void>(unwrap(builder.output("result", result)));
  return MixedFixture{
      unwrap(std::move(builder).finish()),
      input,
      weight,
      product,
      early,
      bias,
      dead,
      result,
  };
}

TK_TEST("Graph arena verification accepts an external-only graph") {
  GraphBuilder builder;
  const ValueId input = unwrap(builder.input("x", f32({1})));
  const std::array<float, 1U> constant_data{{2.0F}};
  const ValueId constant =
      unwrap(builder.constant("c", f32({1}), constant_data));
  static_cast<void>(unwrap(builder.output("input_result", input)));
  static_cast<void>(unwrap(builder.output("constant_result", constant)));
  static_cast<void>(unwrap(builder.output("input_alias", input)));
  const VerifiedGraph graph = unwrap(std::move(builder).finish());

  const std::array<ArenaPlacement, 0U> placements{};
  const GraphArenaLoweringResult result = unwrap(
      GraphArenaPlacementVerifier::verify(graph, placements,
                                          ArenaLimits{0U, 0U}));
  TK_REQUIRE_EQ(result.source_node_count(), 2U);
  TK_REQUIRE_EQ(result.execution_step_count(), 0U);
  TK_REQUIRE(result.values_by_buffer_ordinal().empty());
  TK_REQUIRE(result.requests().empty());
  TK_REQUIRE(!result.buffer_ordinal(input).has_value());
  TK_REQUIRE(!result.buffer_ordinal(constant).has_value());
  TK_REQUIRE(result.allocation_for(input) == nullptr);
  TK_REQUIRE_EQ(result.arena_plan().workspace_bytes(), 0U);
  TK_REQUIRE_EQ(
      result.dump(),
      "tensorkiln.graph_arena_lowering v0 {\n"
      "  source_nodes=2\n"
      "  execution_steps=0\n"
      "  buffers=0\n"
      "}\n"
      "tensorkiln.arena_plan v0 {\n"
      "  alignment_bytes=64\n"
      "  limits {buffers=0, workspace_bytes=0}\n"
      "  stats {buffers=0, payload_bytes=0, reserved_bytes=0, "
      "peak_live_reserved_bytes=0, workspace_bytes=0}\n"
      "}\n");
}

TK_TEST("Graph arena lowering accepts an external-only graph at zero limits") {
  GraphBuilder builder;
  const ValueId input = unwrap(builder.input("x", f32({1})));
  const std::array<float, 1U> constant_data{{2.0F}};
  const ValueId constant =
      unwrap(builder.constant("c", f32({1}), constant_data));
  static_cast<void>(unwrap(builder.output("input_result", input)));
  static_cast<void>(unwrap(builder.output("constant_result", constant)));
  const VerifiedGraph graph = unwrap(std::move(builder).finish());

  const ArenaLimits limits{0U, 0U};
  const GraphArenaLoweringResult lowered =
      unwrap(GraphArenaLowering::run(graph, limits));
  TK_REQUIRE_EQ(lowered.source_node_count(), 2U);
  TK_REQUIRE_EQ(lowered.execution_step_count(), 0U);
  TK_REQUIRE(lowered.values_by_buffer_ordinal().empty());
  TK_REQUIRE(lowered.requests().empty());
  TK_REQUIRE(!lowered.buffer_ordinal(input).has_value());
  TK_REQUIRE(!lowered.buffer_ordinal(constant).has_value());
  TK_REQUIRE_EQ(lowered.arena_plan().workspace_bytes(), 0U);
  require_plan_parity(graph, lowered, limits);
}

TK_TEST("Graph arena verification derives a canonical two-slot chain") {
  const ChainFixture fixture = make_chain();
  const std::string source_dump = fixture.graph.dump();
  const std::array<ArenaPlacement, 3U> placements = chain_placements();
  const GraphArenaLoweringResult result = unwrap(
      GraphArenaPlacementVerifier::verify(fixture.graph, placements));

  TK_REQUIRE_EQ(fixture.graph.dump(), source_dump);
  TK_REQUIRE_EQ(result.source_node_count(), 4U);
  TK_REQUIRE_EQ(result.execution_step_count(), 3U);
  TK_REQUIRE_EQ(result.values_by_buffer_ordinal().size(), 3U);
  TK_REQUIRE_EQ(result.requests().size(), 3U);
  TK_REQUIRE_EQ(result.requests()[0],
                (tensorkiln::ArenaBufferRequest{64U, 0U, 2U}));
  TK_REQUIRE_EQ(result.requests()[1],
                (tensorkiln::ArenaBufferRequest{64U, 1U, 3U}));
  TK_REQUIRE_EQ(result.requests()[2],
                (tensorkiln::ArenaBufferRequest{64U, 2U, 3U}));
  TK_REQUIRE_EQ(require_buffer_ordinal(result, fixture.first), 0U);
  TK_REQUIRE_EQ(require_buffer_ordinal(result, fixture.second), 1U);
  TK_REQUIRE_EQ(require_buffer_ordinal(result, fixture.third), 2U);
  TK_REQUIRE(!result.buffer_ordinal(fixture.input).has_value());
  TK_REQUIRE_EQ(require_value_at(result, 0U), fixture.first);
  TK_REQUIRE_EQ(require_value_at(result, 1U), fixture.second);
  TK_REQUIRE_EQ(require_value_at(result, 2U), fixture.third);
  TK_REQUIRE(result.value_at(3U) == nullptr);
  TK_REQUIRE_EQ(require_allocation(result, fixture.first).offset_bytes(), 0U);
  TK_REQUIRE_EQ(require_allocation(result, fixture.second).offset_bytes(),
                64U);
  TK_REQUIRE_EQ(require_allocation(result, fixture.third).offset_bytes(), 0U);
  TK_REQUIRE_EQ(result.arena_plan().stats().total_payload_bytes, 192U);
  TK_REQUIRE_EQ(result.arena_plan().stats().total_reserved_bytes, 192U);
  TK_REQUIRE_EQ(result.arena_plan().stats().peak_live_reserved_bytes, 128U);
  TK_REQUIRE_EQ(result.arena_plan().workspace_bytes(), 128U);

  TK_REQUIRE_EQ(
      result.dump(),
      "tensorkiln.graph_arena_lowering v0 {\n"
      "  source_nodes=4\n"
      "  execution_steps=3\n"
      "  buffers=3\n"
      "  #b0 <- #n1 %1 step=0\n"
      "  #b1 <- #n2 %2 step=1\n"
      "  #b2 <- #n3 %3 step=2\n"
      "}\n"
      "tensorkiln.arena_plan v0 {\n"
      "  alignment_bytes=64\n"
      "  limits {buffers=4096, workspace_bytes=268435456}\n"
      "  stats {buffers=3, payload_bytes=192, reserved_bytes=192, "
      "peak_live_reserved_bytes=128, workspace_bytes=128}\n"
      "  #b0 offset=0 payload=64 reserved=64 live=[0,2)\n"
      "  #b1 offset=64 payload=64 reserved=64 live=[1,3)\n"
      "  #b2 offset=0 payload=64 reserved=64 live=[2,3)\n"
      "}\n");
}

TK_TEST("Graph arena lowering plans a canonical chain deterministically") {
  const ChainFixture fixture = make_chain();
  const ChainFixture equivalent = make_chain();
  const std::string source_dump = fixture.graph.dump();

  const GraphArenaLoweringResult first =
      unwrap(GraphArenaLowering::run(fixture.graph));
  const GraphArenaLoweringResult repeated =
      unwrap(GraphArenaLowering::run(fixture.graph));
  const GraphArenaLoweringResult separately_owned =
      unwrap(GraphArenaLowering::run(equivalent.graph));

  TK_REQUIRE_EQ(fixture.graph.dump(), source_dump);
  TK_REQUIRE_EQ(first.dump(), repeated.dump());
  TK_REQUIRE_EQ(first.dump(), separately_owned.dump());
  TK_REQUIRE_EQ(first.requests()[0],
                (tensorkiln::ArenaBufferRequest{64U, 0U, 2U}));
  TK_REQUIRE_EQ(first.requests()[1],
                (tensorkiln::ArenaBufferRequest{64U, 1U, 3U}));
  TK_REQUIRE_EQ(first.requests()[2],
                (tensorkiln::ArenaBufferRequest{64U, 2U, 3U}));
  TK_REQUIRE_EQ(require_allocation(first, fixture.first).offset_bytes(), 0U);
  TK_REQUIRE_EQ(require_allocation(first, fixture.second).offset_bytes(),
                64U);
  TK_REQUIRE_EQ(require_allocation(first, fixture.third).offset_bytes(), 0U);
  TK_REQUIRE_EQ(first.arena_plan().workspace_bytes(), 128U);
  require_plan_parity(fixture.graph, first);
}

TK_TEST("Graph arena verification retains outputs and includes dead compute") {
  GraphBuilder builder;
  const ValueId input = unwrap(builder.input("x", f32({16})));
  const ValueId early_output = unwrap(builder.relu(input));
  const ValueId long_lived = unwrap(builder.relu(early_output));
  static_cast<void>(unwrap(builder.output("early", early_output)));
  const ValueId dead = unwrap(builder.relu(input));
  const ValueId result = unwrap(builder.add(long_lived, long_lived));
  static_cast<void>(unwrap(builder.output("result", result)));
  const VerifiedGraph graph = unwrap(std::move(builder).finish());

  const std::array<ArenaPlacement, 4U> placements{{
      {0U, 0U},
      {1U, 64U},
      {2U, 128U},
      {3U, 128U},
  }};
  const GraphArenaLoweringResult lowered =
      unwrap(GraphArenaPlacementVerifier::verify(graph, placements));

  TK_REQUIRE_EQ(lowered.execution_step_count(), 4U);
  TK_REQUIRE_EQ(lowered.requests()[0],
                (tensorkiln::ArenaBufferRequest{64U, 0U, 4U}));
  TK_REQUIRE_EQ(lowered.requests()[1],
                (tensorkiln::ArenaBufferRequest{64U, 1U, 4U}));
  TK_REQUIRE_EQ(lowered.requests()[2],
                (tensorkiln::ArenaBufferRequest{64U, 2U, 3U}));
  TK_REQUIRE_EQ(lowered.requests()[3],
                (tensorkiln::ArenaBufferRequest{64U, 3U, 4U}));
  TK_REQUIRE_EQ(require_buffer_ordinal(lowered, early_output), 0U);
  TK_REQUIRE_EQ(require_buffer_ordinal(lowered, long_lived), 1U);
  TK_REQUIRE_EQ(require_buffer_ordinal(lowered, dead), 2U);
  TK_REQUIRE_EQ(require_buffer_ordinal(lowered, result), 3U);
  TK_REQUIRE_EQ(require_allocation(lowered, dead).offset_bytes(), 128U);
  TK_REQUIRE_EQ(require_allocation(lowered, result).offset_bytes(), 128U);
  TK_REQUIRE_EQ(lowered.arena_plan().stats().peak_live_reserved_bytes, 192U);
  TK_REQUIRE_EQ(lowered.arena_plan().workspace_bytes(), 192U);

  const GraphArenaLoweringResult planned =
      unwrap(GraphArenaLowering::run(graph));
  TK_REQUIRE_EQ(planned.requests()[0],
                (tensorkiln::ArenaBufferRequest{64U, 0U, 4U}));
  TK_REQUIRE_EQ(planned.dump(), lowered.dump());
  require_plan_parity(graph, planned);
}

TK_TEST("Graph arena verification keeps a value through its last fanout use") {
  GraphBuilder builder;
  const ValueId input = unwrap(builder.input("x", f32({1})));
  const ValueId root = unwrap(builder.relu(input));
  const ValueId early_consumer = unwrap(builder.relu(root));
  const ValueId spacer = unwrap(builder.relu(input));
  const ValueId late_consumer = unwrap(builder.add(root, spacer));
  static_cast<void>(unwrap(builder.output("result", late_consumer)));
  const VerifiedGraph graph = unwrap(std::move(builder).finish());

  const std::array<ArenaPlacement, 4U> placements{{
      {0U, 0U},
      {1U, 64U},
      {2U, 128U},
      {3U, 192U},
  }};
  const GraphArenaLoweringResult lowered =
      unwrap(GraphArenaPlacementVerifier::verify(graph, placements));

  TK_REQUIRE_EQ(lowered.requests()[0],
                (tensorkiln::ArenaBufferRequest{4U, 0U, 4U}));
  TK_REQUIRE_EQ(lowered.requests()[1],
                (tensorkiln::ArenaBufferRequest{4U, 1U, 2U}));
  TK_REQUIRE_EQ(lowered.requests()[2],
                (tensorkiln::ArenaBufferRequest{4U, 2U, 4U}));
  TK_REQUIRE_EQ(lowered.requests()[3],
                (tensorkiln::ArenaBufferRequest{4U, 3U, 4U}));
  TK_REQUIRE_EQ(require_buffer_ordinal(lowered, root), 0U);
  TK_REQUIRE_EQ(require_buffer_ordinal(lowered, early_consumer), 1U);
}

TK_TEST("Graph arena verification compacts MatMul steps around constants") {
  GraphBuilder builder;
  const ValueId left = unwrap(builder.input("left", f32({1, 1})));
  const std::array<float, 1U> right_data{{2.0F}};
  const ValueId right =
      unwrap(builder.constant("right", f32({1, 1}), right_data));
  const ValueId product = unwrap(builder.matmul(left, right));
  const std::array<float, 1U> bias_data{{1.0F}};
  const ValueId bias =
      unwrap(builder.constant("bias", f32({1, 1}), bias_data));
  const ValueId result = unwrap(builder.add(product, bias));
  static_cast<void>(unwrap(builder.output("result", result)));
  const VerifiedGraph graph = unwrap(std::move(builder).finish());

  const std::array<ArenaPlacement, 2U> placements{{
      {0U, 0U},
      {1U, 64U},
  }};
  const GraphArenaLoweringResult lowered =
      unwrap(GraphArenaPlacementVerifier::verify(graph, placements));

  TK_REQUIRE_EQ(lowered.source_node_count(), 5U);
  TK_REQUIRE_EQ(lowered.execution_step_count(), 2U);
  TK_REQUIRE_EQ(lowered.requests()[0],
                (tensorkiln::ArenaBufferRequest{4U, 0U, 2U}));
  TK_REQUIRE_EQ(lowered.requests()[1],
                (tensorkiln::ArenaBufferRequest{4U, 1U, 2U}));
  TK_REQUIRE_EQ(require_buffer_ordinal(lowered, product), 0U);
  TK_REQUIRE_EQ(require_buffer_ordinal(lowered, result), 1U);
  TK_REQUIRE(!lowered.buffer_ordinal(left).has_value());
  TK_REQUIRE(!lowered.buffer_ordinal(right).has_value());
  TK_REQUIRE(!lowered.buffer_ordinal(bias).has_value());
  TK_REQUIRE_EQ(lowered.arena_plan().stats().total_payload_bytes, 8U);
  TK_REQUIRE_EQ(lowered.arena_plan().stats().total_reserved_bytes, 128U);
}

TK_TEST("Graph arena lowering compacts constants and preserves MatMul fanout") {
  const MixedFixture fixture = make_mixed_graph();
  const GraphArenaLoweringResult lowered =
      unwrap(GraphArenaLowering::run(fixture.graph));

  TK_REQUIRE_EQ(lowered.source_node_count(), 7U);
  TK_REQUIRE_EQ(lowered.execution_step_count(), 4U);
  TK_REQUIRE_EQ(lowered.requests().size(), 4U);
  TK_REQUIRE_EQ(lowered.requests()[0],
                (tensorkiln::ArenaBufferRequest{4U, 0U, 4U}));
  TK_REQUIRE_EQ(lowered.requests()[1],
                (tensorkiln::ArenaBufferRequest{4U, 1U, 3U}));
  TK_REQUIRE_EQ(lowered.requests()[2],
                (tensorkiln::ArenaBufferRequest{4U, 2U, 3U}));
  TK_REQUIRE_EQ(lowered.requests()[3],
                (tensorkiln::ArenaBufferRequest{4U, 3U, 4U}));
  TK_REQUIRE_EQ(require_buffer_ordinal(lowered, fixture.product), 0U);
  TK_REQUIRE_EQ(require_buffer_ordinal(lowered, fixture.early), 1U);
  TK_REQUIRE_EQ(require_buffer_ordinal(lowered, fixture.dead), 2U);
  TK_REQUIRE_EQ(require_buffer_ordinal(lowered, fixture.result), 3U);
  TK_REQUIRE(!lowered.buffer_ordinal(fixture.input).has_value());
  TK_REQUIRE(!lowered.buffer_ordinal(fixture.weight).has_value());
  TK_REQUIRE(!lowered.buffer_ordinal(fixture.bias).has_value());
  TK_REQUIRE_EQ(require_allocation(lowered, fixture.product).offset_bytes(),
                0U);
  TK_REQUIRE_EQ(require_allocation(lowered, fixture.early).offset_bytes(),
                64U);
  TK_REQUIRE_EQ(require_allocation(lowered, fixture.dead).offset_bytes(),
                128U);
  TK_REQUIRE_EQ(require_allocation(lowered, fixture.result).offset_bytes(),
                64U);
  TK_REQUIRE_EQ(lowered.arena_plan().stats().peak_live_reserved_bytes, 192U);
  TK_REQUIRE_EQ(lowered.arena_plan().workspace_bytes(), 192U);
  require_plan_parity(fixture.graph, lowered);
}

TK_TEST("Graph arena lowering preserves exact arena limits") {
  const ChainFixture fixture = make_chain();
  const ArenaLimits exact_limits{3U, 128U};
  const GraphArenaLoweringResult exact =
      unwrap(GraphArenaLowering::run(fixture.graph, exact_limits));
  TK_REQUIRE_EQ(exact.arena_plan().limits(), exact_limits);
  TK_REQUIRE_EQ(exact.arena_plan().workspace_bytes(), 128U);

  const auto bad_buffer_limit =
      GraphArenaLowering::run(fixture.graph, ArenaLimits{2U, 128U});
  TK_REQUIRE_EQ(
      require_error(bad_buffer_limit,
                    ErrorCode::arena_buffer_limit_exceeded)
          .message,
      "arena has 3 buffer requests; limit is 2");

  const auto bad_workspace_limit =
      GraphArenaLowering::run(fixture.graph, ArenaLimits{3U, 127U});
  TK_REQUIRE_EQ(
      require_error(bad_workspace_limit,
                    ErrorCode::arena_workspace_limit_exceeded)
          .message,
      "arena workspace requires 128 bytes; limit is 127");
}

TK_TEST("Graph arena lowering reflects explicit DCE pass ordering") {
  const MixedFixture fixture = make_mixed_graph();
  const GraphArenaLoweringResult before =
      unwrap(GraphArenaLowering::run(fixture.graph));
  const tensorkiln::DeadCodeEliminationResult eliminated =
      unwrap(tensorkiln::DeadCodeElimination::run(fixture.graph));
  const GraphArenaLoweringResult after =
      unwrap(GraphArenaLowering::run(eliminated.graph()));

  TK_REQUIRE_EQ(before.source_node_count(), 7U);
  TK_REQUIRE_EQ(before.requests().size(), 4U);
  TK_REQUIRE_EQ(before.arena_plan().workspace_bytes(), 192U);
  TK_REQUIRE_EQ(after.source_node_count(), 5U);
  TK_REQUIRE_EQ(after.requests().size(), 2U);
  TK_REQUIRE_EQ(after.arena_plan().workspace_bytes(), 128U);
  TK_REQUIRE(eliminated.provenance().for_source(fixture.dead) == nullptr);

  const tensorkiln::NodeProvenance* result_lineage =
      eliminated.provenance().for_source(fixture.result);
  TK_REQUIRE(result_lineage != nullptr);
  TK_REQUIRE_EQ(require_buffer_ordinal(after, result_lineage->result_value()),
                1U);
  TK_REQUIRE(!after.buffer_ordinal(fixture.result).has_value());
  require_plan_parity(eliminated.graph(), after);
}

TK_TEST("Graph arena lowering agreement rejects altered forward evidence") {
  const ChainFixture fixture = make_chain();
  const GraphArenaLoweringResult lowered =
      unwrap(GraphArenaLowering::run(fixture.graph));

  std::vector<ValueId> altered_values(
      lowered.values_by_buffer_ordinal().begin(),
      lowered.values_by_buffer_ordinal().end());
  std::swap(altered_values[0], altered_values[1]);
  const auto bad_mapping =
      tensorkiln::detail::verify_graph_arena_lowering_agreement(
          fixture.graph, altered_values, lowered.requests(),
          lowered.execution_step_count(), lowered.arena_plan(),
          lowered.arena_plan().limits());
  TK_REQUIRE_EQ(
      require_error(bad_mapping, ErrorCode::compiler_internal_invariant)
          .message,
      "graph arena forward and reverse value mappings disagree at #b0");

  std::vector<tensorkiln::ArenaBufferRequest> altered_requests(
      lowered.requests().begin(), lowered.requests().end());
  altered_requests[0].live_end_step_exclusive = 3U;
  const auto bad_request =
      tensorkiln::detail::verify_graph_arena_lowering_agreement(
          fixture.graph, lowered.values_by_buffer_ordinal(),
          altered_requests, lowered.execution_step_count(),
          lowered.arena_plan(), lowered.arena_plan().limits());
  TK_REQUIRE_EQ(
      require_error(bad_request, ErrorCode::compiler_internal_invariant)
          .message,
      "graph arena forward and reverse requests disagree at #b0");

  const std::vector<tensorkiln::ArenaBufferRequest> disjoint_requests{
      {64U, 0U, 1U},
      {64U, 1U, 2U},
      {64U, 2U, 3U},
  };
  const tensorkiln::ArenaPlan unsafe_for_graph =
      unwrap(ArenaPlanner::run(disjoint_requests));
  const auto bad_placements =
      tensorkiln::detail::verify_graph_arena_lowering_agreement(
          fixture.graph, lowered.values_by_buffer_ordinal(),
          disjoint_requests, lowered.execution_step_count(),
          unsafe_for_graph, unsafe_for_graph.limits());
  TK_REQUIRE_EQ(
      require_error(bad_placements, ErrorCode::compiler_internal_invariant)
          .message,
      "graph arena lowering failed reverse verification: arena_live_overlap: "
      "arena buffers #b0 and #b1 overlap in bytes [0,64) and [0,64) while "
      "lifetimes [0,2) and [1,3) intersect");
}

TK_TEST("Graph arena verification preserves arena diagnostic precedence") {
  const ChainFixture fixture = make_chain();
  const std::array<ArenaPlacement, 3U> overlap{{
      {0U, 0U},
      {1U, 0U},
      {2U, 64U},
  }};
  const auto bad_overlap =
      GraphArenaPlacementVerifier::verify(fixture.graph, overlap);
  TK_REQUIRE_EQ(
      require_error(bad_overlap, ErrorCode::arena_live_overlap).message,
      "arena buffers #b0 and #b1 overlap in bytes [0,64) and [0,64) while "
      "lifetimes [0,2) and [1,3) intersect");

  const std::array<ArenaPlacement, 3U> unaligned{{
      {0U, 0U},
      {1U, 1U},
      {2U, 0U},
  }};
  const auto bad_alignment =
      GraphArenaPlacementVerifier::verify(fixture.graph, unaligned);
  TK_REQUIRE_EQ(
      require_error(bad_alignment, ErrorCode::arena_alignment_invalid)
          .message,
      "arena buffer #b1 offset 1 is not 64-byte aligned");

  const std::array<ArenaPlacement, 2U> too_few{{
      {0U, 0U},
      {1U, 64U},
  }};
  const auto bad_count =
      GraphArenaPlacementVerifier::verify(fixture.graph, too_few);
  TK_REQUIRE_EQ(
      require_error(bad_count,
                    ErrorCode::arena_placement_count_mismatch)
          .message,
      "arena plan has 2 placements; expected 3");

  const std::array<ArenaPlacement, 3U> unknown{{
      {7U, 0U},
      {1U, 64U},
      {2U, 0U},
  }};
  const auto bad_unknown =
      GraphArenaPlacementVerifier::verify(fixture.graph, unknown);
  TK_REQUIRE_EQ(
      require_error(bad_unknown,
                    ErrorCode::arena_placement_buffer_not_found)
          .message,
      "arena placement 0 references unknown buffer #b7");

  const std::array<ArenaPlacement, 3U> duplicate{{
      {0U, 0U},
      {0U, 64U},
      {2U, 0U},
  }};
  const auto bad_duplicate =
      GraphArenaPlacementVerifier::verify(fixture.graph, duplicate);
  TK_REQUIRE_EQ(
      require_error(bad_duplicate, ErrorCode::arena_placement_duplicate)
          .message,
      "arena buffer #b0 has multiple placements");

  const std::array<ArenaPlacement, 3U> placements = chain_placements();
  const auto bad_buffer_limit = GraphArenaPlacementVerifier::verify(
      fixture.graph, too_few, ArenaLimits{2U, 128U});
  TK_REQUIRE_EQ(
      require_error(bad_buffer_limit,
                    ErrorCode::arena_buffer_limit_exceeded)
          .message,
      "arena has 3 buffer requests; limit is 2");

  const auto bad_workspace_limit = GraphArenaPlacementVerifier::verify(
      fixture.graph, placements, ArenaLimits{3U, 127U});
  TK_REQUIRE_EQ(
      require_error(bad_workspace_limit,
                    ErrorCode::arena_workspace_limit_exceeded)
          .message,
      "arena workspace requires 128 bytes; limit is 127");

  const GraphArenaLoweringResult exact = unwrap(
      GraphArenaPlacementVerifier::verify(fixture.graph, placements,
                                          ArenaLimits{3U, 128U}));
  TK_REQUIRE_EQ(exact.arena_plan().limits(), (ArenaLimits{3U, 128U}));
}

struct DetachedLowering final {
  GraphArenaLoweringResult result;
  ValueId external;
  ValueId computed;
};

[[nodiscard]] DetachedLowering make_detached_lowering() {
  const ChainFixture fixture = make_chain();
  const std::array<ArenaPlacement, 3U> placements = chain_placements();
  GraphArenaLoweringResult result = unwrap(
      GraphArenaPlacementVerifier::verify(fixture.graph, placements));
  return DetachedLowering{
      std::move(result),
      fixture.input,
      fixture.third,
  };
}

TK_TEST("Graph arena results own mappings and reject foreign handles") {
  DetachedLowering detached = make_detached_lowering();
  TK_REQUIRE(!detached.result.buffer_ordinal(detached.external).has_value());
  TK_REQUIRE_EQ(require_buffer_ordinal(detached.result, detached.computed),
                2U);
  TK_REQUIRE_EQ(
      require_allocation(detached.result, detached.computed).offset_bytes(),
      0U);

  const ChainFixture foreign = make_chain();
  TK_REQUIRE_EQ(foreign.third.ordinal(), detached.computed.ordinal());
  TK_REQUIRE_EQ(foreign.input.ordinal(), detached.external.ordinal());
  TK_REQUIRE(!detached.result.buffer_ordinal(foreign.input).has_value());
  TK_REQUIRE(detached.result.allocation_for(foreign.input) == nullptr);
  TK_REQUIRE(!detached.result.buffer_ordinal(foreign.third).has_value());
  TK_REQUIRE(detached.result.allocation_for(foreign.third) == nullptr);

  GraphArenaLoweringResult moved = std::move(detached.result);
  TK_REQUIRE_EQ(require_buffer_ordinal(moved, detached.computed), 2U);
  TK_REQUIRE_EQ(require_value_at(moved, 2U), detached.computed);
}

}  // namespace
