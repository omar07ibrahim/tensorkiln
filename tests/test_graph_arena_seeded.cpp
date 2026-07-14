#include "test.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "tensorkiln/graph_arena.hpp"

namespace {

using tensorkiln::AddOp;
using tensorkiln::ArenaAllocation;
using tensorkiln::ArenaBufferRequest;
using tensorkiln::ArenaLimits;
using tensorkiln::Diagnostic;
using tensorkiln::ErrorCode;
using tensorkiln::GraphArenaLowering;
using tensorkiln::GraphArenaLoweringResult;
using tensorkiln::GraphBuilder;
using tensorkiln::GraphLimits;
using tensorkiln::ReluOp;
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

[[nodiscard]] TensorType f32(
    const std::span<const std::int64_t> extents) {
  return unwrap(TensorType::create(unwrap(Shape::create(extents))));
}

[[nodiscard]] TensorType f32(
    const std::initializer_list<std::int64_t> extents) {
  return f32(std::span<const std::int64_t>{extents.begin(), extents.size()});
}

[[nodiscard]] TensorType scalar_f32() {
  return unwrap(TensorType::create(Shape::scalar()));
}

class DeterministicGenerator final {
 public:
  explicit DeterministicGenerator(const std::uint32_t seed) noexcept
      : state_(static_cast<std::uint64_t>(seed)) {}

  [[nodiscard]] std::uint64_t next() noexcept {
    state_ += UINT64_C(0x9e3779b97f4a7c15);
    std::uint64_t value = state_;
    value = (value ^ (value >> 30U)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27U)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31U);
  }

  [[nodiscard]] std::size_t index(const std::size_t bound) {
    TK_REQUIRE(bound > 0U);
    const std::uint64_t range = static_cast<std::uint64_t>(bound);
    const std::uint64_t threshold = (UINT64_C(0) - range) % range;
    std::uint64_t value = next();
    while (value < threshold) {
      value = next();
    }
    return static_cast<std::size_t>(value % range);
  }

 private:
  std::uint64_t state_;
};

struct SeededFixture final {
  VerifiedGraph graph;
  std::vector<ValueId> compute_values;
  std::vector<ValueId> external_values;
};

[[nodiscard]] SeededFixture make_seeded_fixture(
    const std::uint32_t seed) {
  constexpr std::size_t compute_count = 64U;
  DeterministicGenerator generator(seed);
  GraphBuilder builder;
  const TensorType type = scalar_f32();

  std::vector<ValueId> external_values;
  std::vector<ValueId> all_values;
  external_values.reserve(4U);
  all_values.reserve(compute_count + 4U);
  const ValueId left = unwrap(builder.input("left", type));
  const ValueId right = unwrap(builder.input("right", type));
  const std::array<float, 1U> base_data{{0.25F}};
  const ValueId base = unwrap(builder.constant("base", type, base_data));
  external_values.push_back(left);
  external_values.push_back(right);
  external_values.push_back(base);
  all_values.push_back(left);
  all_values.push_back(right);
  all_values.push_back(base);

  std::vector<ValueId> compute_values;
  compute_values.reserve(compute_count);
  const ValueId anchor = unwrap(builder.relu(left));
  compute_values.push_back(anchor);
  all_values.push_back(anchor);

  for (std::size_t step = 1U; step < compute_count; ++step) {
    if (step == compute_count / 2U) {
      const std::array<float, 1U> mid_data{{-0.5F}};
      const ValueId mid =
          unwrap(builder.constant("mid", type, mid_data));
      external_values.push_back(mid);
      all_values.push_back(mid);
    }

    ValueId result = right;
    if (step == compute_count - 1U) {
      result = unwrap(builder.relu(right));
    } else if (step == 17U || step == 33U || step == 49U) {
      const ValueId other = all_values[generator.index(all_values.size())];
      result = unwrap(builder.add(anchor, other));
    } else if ((step % 11U) == 0U) {
      const ValueId repeated =
          all_values[generator.index(all_values.size())];
      result = unwrap(builder.add(repeated, repeated));
    } else if (generator.index(3U) == 0U) {
      const ValueId operand =
          all_values[generator.index(all_values.size())];
      result = unwrap(builder.relu(operand));
    } else {
      const ValueId first = all_values[generator.index(all_values.size())];
      const ValueId second = all_values[generator.index(all_values.size())];
      result = unwrap(builder.add(first, second));
    }
    compute_values.push_back(result);
    all_values.push_back(result);
  }

  static_cast<void>(unwrap(builder.output("early", compute_values[1U])));
  static_cast<void>(
      unwrap(builder.output("early_alias", compute_values[1U])));
  static_cast<void>(unwrap(
      builder.output("middle", compute_values[compute_count / 2U])));
  static_cast<void>(
      unwrap(builder.output("late", compute_values[compute_count - 2U])));

  return SeededFixture{
      unwrap(std::move(builder).finish()),
      std::move(compute_values),
      std::move(external_values),
  };
}

[[nodiscard]] std::vector<ArenaBufferRequest> brute_force_requests(
    const SeededFixture& fixture) {
  const std::size_t compute_count = fixture.compute_values.size();
  TK_REQUIRE(compute_count <=
             static_cast<std::size_t>(UINT32_MAX));
  const auto terminal_step = static_cast<std::uint32_t>(compute_count);
  std::vector<ArenaBufferRequest> requests;
  requests.reserve(compute_count);

  for (std::size_t producer_step = 0U;
       producer_step < compute_count; ++producer_step) {
    const ValueId value = fixture.compute_values[producer_step];
    const std::size_t source_ordinal =
        static_cast<std::size_t>(value.ordinal());
    TK_REQUIRE(source_ordinal < fixture.graph.nodes().size());
    const tensorkiln::Node& producer =
        fixture.graph.nodes()[source_ordinal];
    TK_REQUIRE_EQ(producer.output(), value);
    TK_REQUIRE(std::holds_alternative<AddOp>(producer.operation()) ||
               std::holds_alternative<ReluOp>(producer.operation()));

    const auto begin = static_cast<std::uint32_t>(producer_step);
    auto end = static_cast<std::uint32_t>(producer_step + 1U);
    for (std::size_t consumer_step = 0U;
         consumer_step < compute_count; ++consumer_step) {
      const ValueId consumer_value = fixture.compute_values[consumer_step];
      const tensorkiln::Node& consumer = fixture.graph.nodes()[
          static_cast<std::size_t>(consumer_value.ordinal())];
      for (const ValueId operand : consumer.inputs()) {
        if (operand == value) {
          end = std::max(
              end, static_cast<std::uint32_t>(consumer_step + 1U));
        }
      }
    }
    for (const tensorkiln::GraphOutput& output : fixture.graph.outputs()) {
      if (output.value() == value) {
        end = terminal_step;
      }
    }
    requests.push_back(ArenaBufferRequest{
        static_cast<std::uint64_t>(producer.output_type().byte_count()),
        begin,
        end,
    });
  }
  return requests;
}

// Generated fixtures stay far below uint64 limits, so direct arithmetic keeps
// this test oracle visibly independent from the production overflow helpers.
[[nodiscard]] std::uint64_t reserved_bytes(
    const ArenaBufferRequest& request) noexcept {
  return (request.size_bytes + (tensorkiln::kArenaAlignmentBytes - 1U)) &
         ~(tensorkiln::kArenaAlignmentBytes - 1U);
}

[[nodiscard]] bool lifetimes_overlap(
    const ArenaBufferRequest& left,
    const ArenaBufferRequest& right) noexcept {
  return left.live_begin_step < right.live_end_step_exclusive &&
         right.live_begin_step < left.live_end_step_exclusive;
}

[[nodiscard]] bool byte_ranges_overlap(
    const ArenaAllocation& left,
    const ArenaAllocation& right) noexcept {
  return left.offset_bytes() <
             right.offset_bytes() + right.reserved_bytes() &&
         right.offset_bytes() <
             left.offset_bytes() + left.reserved_bytes();
}

void require_matches_brute_force_oracle(
    const SeededFixture& fixture,
    const std::vector<ArenaBufferRequest>& oracle,
    const GraphArenaLoweringResult& lowered) {
  TK_REQUIRE_EQ(lowered.execution_step_count(), oracle.size());
  TK_REQUIRE_EQ(lowered.values_by_buffer_ordinal().size(), oracle.size());
  TK_REQUIRE_EQ(lowered.requests().size(), oracle.size());
  TK_REQUIRE_EQ(lowered.arena_plan().allocations().size(), oracle.size());

  std::uint64_t total_payload_bytes = 0U;
  std::uint64_t total_reserved_bytes = 0U;
  std::uint64_t workspace_bytes = 0U;
  for (std::size_t index = 0U; index < oracle.size(); ++index) {
    const auto ordinal = static_cast<std::uint32_t>(index);
    TK_REQUIRE_EQ(lowered.values_by_buffer_ordinal()[index],
                  fixture.compute_values[index]);
    TK_REQUIRE_EQ(lowered.requests()[index], oracle[index]);
    TK_REQUIRE_EQ(lowered.buffer_ordinal(fixture.compute_values[index]),
                  ordinal);
    const ArenaAllocation* allocation =
        lowered.arena_plan().allocation_at(ordinal);
    TK_REQUIRE(allocation != nullptr);
    TK_REQUIRE_EQ(allocation->buffer_ordinal(), ordinal);
    TK_REQUIRE_EQ(allocation->payload_bytes(), oracle[index].size_bytes);
    TK_REQUIRE_EQ(allocation->reserved_bytes(), reserved_bytes(oracle[index]));
    TK_REQUIRE_EQ(allocation->live_begin_step(),
                  oracle[index].live_begin_step);
    TK_REQUIRE_EQ(allocation->live_end_step_exclusive(),
                  oracle[index].live_end_step_exclusive);
    TK_REQUIRE_EQ(allocation->offset_bytes() %
                      tensorkiln::kArenaAlignmentBytes,
                  0U);
    total_payload_bytes += oracle[index].size_bytes;
    total_reserved_bytes += reserved_bytes(oracle[index]);
    workspace_bytes =
        std::max(workspace_bytes,
                 allocation->offset_bytes() + allocation->reserved_bytes());
  }
  for (const ValueId external : fixture.external_values) {
    TK_REQUIRE(!lowered.buffer_ordinal(external).has_value());
    TK_REQUIRE(lowered.allocation_for(external) == nullptr);
  }

  for (std::size_t left = 0U; left < oracle.size(); ++left) {
    for (std::size_t right = left + 1U; right < oracle.size(); ++right) {
      if (!lifetimes_overlap(oracle[left], oracle[right])) {
        continue;
      }
      const ArenaAllocation* left_allocation =
          lowered.arena_plan().allocation_at(
              static_cast<std::uint32_t>(left));
      const ArenaAllocation* right_allocation =
          lowered.arena_plan().allocation_at(
              static_cast<std::uint32_t>(right));
      TK_REQUIRE(left_allocation != nullptr);
      TK_REQUIRE(right_allocation != nullptr);
      TK_REQUIRE(!byte_ranges_overlap(*left_allocation, *right_allocation));
    }
  }

  std::uint64_t peak_live_reserved_bytes = 0U;
  for (std::uint32_t step = 0U;
       step < lowered.execution_step_count(); ++step) {
    std::uint64_t live_reserved_bytes = 0U;
    for (const ArenaBufferRequest& request : oracle) {
      if (request.live_begin_step <= step &&
          step < request.live_end_step_exclusive) {
        live_reserved_bytes += reserved_bytes(request);
      }
    }
    peak_live_reserved_bytes =
        std::max(peak_live_reserved_bytes, live_reserved_bytes);
  }

  TK_REQUIRE_EQ(lowered.arena_plan().stats().buffer_count,
                static_cast<std::uint32_t>(oracle.size()));
  TK_REQUIRE_EQ(lowered.arena_plan().stats().total_payload_bytes,
                total_payload_bytes);
  TK_REQUIRE_EQ(lowered.arena_plan().stats().total_reserved_bytes,
                total_reserved_bytes);
  TK_REQUIRE_EQ(lowered.arena_plan().stats().peak_live_reserved_bytes,
                peak_live_reserved_bytes);
  TK_REQUIRE_EQ(lowered.arena_plan().workspace_bytes(), workspace_bytes);
  TK_REQUIRE(peak_live_reserved_bytes <= workspace_bytes);
  TK_REQUIRE(workspace_bytes <= total_reserved_bytes);
}

TK_TEST("Graph arena lowering agrees with a seeded brute-force DAG oracle") {
  std::set<std::string> graph_signatures;
  std::set<std::string> placement_signatures;
  std::size_t plans_with_reuse = 0U;

  for (std::uint32_t seed = 1U; seed <= 64U; ++seed) {
    const SeededFixture fixture = make_seeded_fixture(seed);
    const SeededFixture rebuilt = make_seeded_fixture(seed);
    const std::vector<ArenaBufferRequest> oracle =
        brute_force_requests(fixture);
    const GraphArenaLoweringResult first =
        unwrap(GraphArenaLowering::run(fixture.graph));
    const GraphArenaLoweringResult repeated =
        unwrap(GraphArenaLowering::run(fixture.graph));
    const GraphArenaLoweringResult rebuilt_lowering =
        unwrap(GraphArenaLowering::run(rebuilt.graph));

    require_matches_brute_force_oracle(fixture, oracle, first);
    TK_REQUIRE_EQ(fixture.graph.dump(), rebuilt.graph.dump());
    TK_REQUIRE_EQ(first.dump(), repeated.dump());
    TK_REQUIRE_EQ(first.dump(), rebuilt_lowering.dump());
    TK_REQUIRE_EQ(oracle[1U].live_end_step_exclusive, 64U);
    TK_REQUIRE(oracle[0U].live_end_step_exclusive >= 50U);
    TK_REQUIRE_EQ(
        fixture.compute_values[32U].ordinal() -
            fixture.compute_values[31U].ordinal(),
        2U);
    TK_REQUIRE(fixture.graph.outputs().back().value() !=
               fixture.compute_values.back());

    graph_signatures.insert(fixture.graph.dump());
    std::string placement_signature =
        std::to_string(first.arena_plan().workspace_bytes()) + ":";
    for (const ArenaAllocation& allocation :
         first.arena_plan().allocations()) {
      placement_signature +=
          std::to_string(allocation.offset_bytes()) + ",";
    }
    placement_signatures.insert(std::move(placement_signature));
    if (first.arena_plan().workspace_bytes() <
        first.arena_plan().stats().total_reserved_bytes) {
      ++plans_with_reuse;
    }
  }

  TK_REQUIRE(graph_signatures.size() >= 60U);
  TK_REQUIRE(placement_signatures.size() >= 16U);
  TK_REQUIRE_EQ(plans_with_reuse, 64U);
}

struct MatrixCase final {
  std::string_view name;
  std::vector<std::int64_t> input_shape;
  std::vector<std::int64_t> first_weight_shape;
  std::vector<std::int64_t> expand_shape;
  std::vector<std::int64_t> second_weight_shape;
  std::array<std::uint64_t, 5U> payload_bytes;
  std::array<std::uint64_t, 5U> offsets;
  std::uint64_t peak_live_reserved_bytes;
  std::uint64_t workspace_bytes;
};

struct MatrixFixture final {
  VerifiedGraph graph;
  std::vector<ValueId> external_values;
  std::array<ValueId, 5U> compute_values;
};

[[nodiscard]] std::vector<float> constant_data(
    const TensorType& type, const float value) {
  return std::vector<float>(static_cast<std::size_t>(type.numel()), value);
}

[[nodiscard]] MatrixFixture make_matrix_fixture(const MatrixCase& test_case) {
  GraphBuilder builder;
  const TensorType input_type = f32(test_case.input_shape);
  const TensorType first_weight_type = f32(test_case.first_weight_shape);
  const TensorType expand_type = f32(test_case.expand_shape);
  const TensorType second_weight_type = f32(test_case.second_weight_shape);
  const ValueId input = unwrap(builder.input("input", input_type));
  const std::vector<float> first_weight_data =
      constant_data(first_weight_type, 0.25F);
  const ValueId first_weight = unwrap(builder.constant(
      "first_weight", first_weight_type, first_weight_data));
  const ValueId first_product =
      unwrap(builder.matmul(input, first_weight));
  const ValueId activated = unwrap(builder.relu(first_product));
  const std::vector<float> expand_data =
      constant_data(expand_type, -0.5F);
  const ValueId expand =
      unwrap(builder.constant("expand", expand_type, expand_data));
  const ValueId dead = unwrap(builder.add(activated, expand));
  const std::vector<float> second_weight_data =
      constant_data(second_weight_type, 0.125F);
  const ValueId second_weight = unwrap(builder.constant(
      "second_weight", second_weight_type, second_weight_data));
  const ValueId second_product =
      unwrap(builder.matmul(first_product, second_weight));
  const ValueId result = unwrap(builder.relu(second_product));
  static_cast<void>(unwrap(builder.output("result", result)));
  return MatrixFixture{
      unwrap(std::move(builder).finish()),
      {input, first_weight, expand, second_weight},
      {first_product, activated, dead, second_product, result},
  };
}

TK_TEST("Graph arena lowering covers heterogeneous MatMul payload boundaries") {
  const std::vector<MatrixCase> cases{
      {"small heterogeneous",
       {2, 3}, {3, 2}, {5, 1, 1}, {2, 5},
       {16U, 16U, 80U, 40U, 40U},
       {0U, 64U, 128U, 64U, 0U}, 256U, 256U},
      {"64-byte crossing",
       {1, 3}, {3, 15}, {2, 1, 1}, {15, 17},
       {60U, 60U, 120U, 68U, 68U},
       {0U, 64U, 128U, 64U, 192U}, 256U, 320U},
      {"larger shrink",
       {3, 4}, {4, 6}, {2, 1, 1}, {6, 3},
       {72U, 72U, 144U, 36U, 36U},
       {0U, 128U, 256U, 128U, 0U}, 448U, 448U},
      {"exact-64 output",
       {1, 8}, {8, 8}, {9, 1, 1}, {8, 16},
       {32U, 32U, 288U, 64U, 64U},
       {0U, 64U, 128U, 64U, 0U}, 448U, 448U},
  };
  const std::array<std::pair<std::uint32_t, std::uint32_t>, 5U> lifetimes{{
      {0U, 4U},
      {1U, 3U},
      {2U, 3U},
      {3U, 5U},
      {4U, 5U},
  }};

  for (const MatrixCase& test_case : cases) {
    const MatrixFixture fixture = make_matrix_fixture(test_case);
    const GraphArenaLoweringResult lowered =
        unwrap(GraphArenaLowering::run(fixture.graph));
    TK_REQUIRE_EQ(lowered.source_node_count(), 9U);
    TK_REQUIRE_EQ(lowered.execution_step_count(), 5U);
    TK_REQUIRE_EQ(lowered.requests().size(), 5U);
    std::uint64_t total_payload_bytes = 0U;
    std::uint64_t total_reserved_bytes = 0U;
    for (std::size_t index = 0U; index < test_case.payload_bytes.size();
         ++index) {
      const ArenaBufferRequest expected{
          test_case.payload_bytes[index],
          lifetimes[index].first,
          lifetimes[index].second,
      };
      TK_REQUIRE_EQ(lowered.values_by_buffer_ordinal()[index],
                    fixture.compute_values[index]);
      TK_REQUIRE_EQ(lowered.requests()[index], expected);
      TK_REQUIRE_EQ(lowered.buffer_ordinal(fixture.compute_values[index]),
                    static_cast<std::uint32_t>(index));
      const ArenaAllocation* allocation = lowered.arena_plan().allocation_at(
          static_cast<std::uint32_t>(index));
      TK_REQUIRE(allocation != nullptr);
      TK_REQUIRE_EQ(allocation->offset_bytes(), test_case.offsets[index]);
      TK_REQUIRE_EQ(allocation->reserved_bytes(), reserved_bytes(expected));
      total_payload_bytes += expected.size_bytes;
      total_reserved_bytes += reserved_bytes(expected);
    }
    for (const ValueId external : fixture.external_values) {
      TK_REQUIRE(!lowered.buffer_ordinal(external).has_value());
    }
    TK_REQUIRE_EQ(lowered.arena_plan().stats().total_payload_bytes,
                  total_payload_bytes);
    TK_REQUIRE_EQ(lowered.arena_plan().stats().total_reserved_bytes,
                  total_reserved_bytes);
    TK_REQUIRE_EQ(lowered.arena_plan().stats().peak_live_reserved_bytes,
                  test_case.peak_live_reserved_bytes);
    TK_REQUIRE_EQ(lowered.arena_plan().workspace_bytes(),
                  test_case.workspace_bytes);

    if (test_case.name == "64-byte crossing") {
      const MatrixFixture rebuilt = make_matrix_fixture(test_case);
      const GraphArenaLoweringResult rebuilt_lowering =
          unwrap(GraphArenaLowering::run(rebuilt.graph));
      TK_REQUIRE_EQ(rebuilt.graph.dump(), fixture.graph.dump());
      TK_REQUIRE_EQ(rebuilt_lowering.dump(), lowered.dump());
    }
  }
}

TK_TEST("Graph arena lowering covers rank-four batched MatMul lifetimes") {
  GraphBuilder builder;
  const TensorType left_type = f32({2, 1, 2, 3});
  const TensorType right_type = f32({1, 3, 3, 4});
  const TensorType projection_type = f32({1, 3, 4, 1});
  const TensorType bias_type = f32({2, 1});
  const ValueId left = unwrap(builder.input("left", left_type));
  const std::vector<float> right_data = constant_data(right_type, 0.25F);
  const ValueId right =
      unwrap(builder.constant("right", right_type, right_data));
  const ValueId wide = unwrap(builder.matmul(left, right));
  const std::vector<float> projection_data =
      constant_data(projection_type, -0.125F);
  const ValueId projection = unwrap(builder.constant(
      "projection", projection_type, projection_data));
  const ValueId narrow = unwrap(builder.matmul(wide, projection));
  const std::vector<float> bias_data = constant_data(bias_type, 0.5F);
  const ValueId bias =
      unwrap(builder.constant("bias", bias_type, bias_data));
  const ValueId result = unwrap(builder.add(narrow, bias));
  static_cast<void>(unwrap(builder.output("wide", wide)));
  static_cast<void>(unwrap(builder.output("result", result)));
  const VerifiedGraph graph = unwrap(std::move(builder).finish());

  const GraphArenaLoweringResult lowered =
      unwrap(GraphArenaLowering::run(graph));
  const std::array<ValueId, 3U> values{{wide, narrow, result}};
  const std::array<ArenaBufferRequest, 3U> expected{{
      {192U, 0U, 3U},
      {48U, 1U, 3U},
      {48U, 2U, 3U},
  }};
  const std::array<std::uint64_t, 3U> offsets{{0U, 192U, 256U}};
  TK_REQUIRE_EQ(lowered.source_node_count(), 7U);
  TK_REQUIRE_EQ(lowered.execution_step_count(), 3U);
  for (std::size_t index = 0U; index < values.size(); ++index) {
    TK_REQUIRE_EQ(lowered.values_by_buffer_ordinal()[index], values[index]);
    TK_REQUIRE_EQ(lowered.requests()[index], expected[index]);
    const ArenaAllocation* allocation = lowered.arena_plan().allocation_at(
        static_cast<std::uint32_t>(index));
    TK_REQUIRE(allocation != nullptr);
    TK_REQUIRE_EQ(allocation->offset_bytes(), offsets[index]);
  }
  TK_REQUIRE(!lowered.buffer_ordinal(left).has_value());
  TK_REQUIRE(!lowered.buffer_ordinal(right).has_value());
  TK_REQUIRE(!lowered.buffer_ordinal(projection).has_value());
  TK_REQUIRE(!lowered.buffer_ordinal(bias).has_value());
  TK_REQUIRE_EQ(lowered.arena_plan().stats().total_payload_bytes, 288U);
  TK_REQUIRE_EQ(lowered.arena_plan().stats().total_reserved_bytes, 320U);
  TK_REQUIRE_EQ(lowered.arena_plan().stats().peak_live_reserved_bytes, 320U);
  TK_REQUIRE_EQ(lowered.arena_plan().workspace_bytes(), 320U);
}

struct ScaleChainFixture final {
  VerifiedGraph graph;
  ValueId input;
  ValueId result;
};

[[nodiscard]] ScaleChainFixture make_scale_chain(
    const std::uint32_t compute_count) {
  GraphLimits limits;
  limits.max_nodes = compute_count + 1U;
  limits.max_outputs = 1U;
  GraphBuilder builder(limits);
  const ValueId input = unwrap(builder.input("input", f32({1})));
  ValueId current = input;
  for (std::uint32_t index = 0U; index < compute_count; ++index) {
    current = unwrap(builder.relu(current));
  }
  static_cast<void>(unwrap(builder.output("result", current)));
  return ScaleChainFixture{
      unwrap(std::move(builder).finish()),
      input,
      current,
  };
}

TK_TEST("Graph arena lowering accepts the exact default buffer boundary") {
  constexpr std::uint32_t compute_count =
      tensorkiln::kDefaultMaxArenaBuffers;
  const ScaleChainFixture fixture = make_scale_chain(compute_count);
  const GraphArenaLoweringResult lowered =
      unwrap(GraphArenaLowering::run(fixture.graph));

  TK_REQUIRE_EQ(fixture.graph.limits().max_nodes, compute_count + 1U);
  TK_REQUIRE_EQ(lowered.source_node_count(), compute_count + 1U);
  TK_REQUIRE_EQ(lowered.execution_step_count(), compute_count);
  TK_REQUIRE_EQ(lowered.requests().size(), compute_count);
  TK_REQUIRE_EQ(lowered.arena_plan().allocations().size(), compute_count);
  TK_REQUIRE(!lowered.buffer_ordinal(fixture.input).has_value());
  TK_REQUIRE_EQ(lowered.buffer_ordinal(fixture.result), compute_count - 1U);
  TK_REQUIRE_EQ(lowered.requests()[0U],
                (ArenaBufferRequest{4U, 0U, 2U}));
  TK_REQUIRE_EQ(lowered.requests()[compute_count - 2U],
                (ArenaBufferRequest{4U, compute_count - 2U,
                                    compute_count}));
  TK_REQUIRE_EQ(lowered.requests()[compute_count - 1U],
                (ArenaBufferRequest{4U, compute_count - 1U,
                                    compute_count}));
  TK_REQUIRE_EQ(lowered.arena_plan().stats().buffer_count, compute_count);
  TK_REQUIRE_EQ(lowered.arena_plan().stats().total_payload_bytes, 16384U);
  TK_REQUIRE_EQ(lowered.arena_plan().stats().total_reserved_bytes, 262144U);
  TK_REQUIRE_EQ(lowered.arena_plan().stats().peak_live_reserved_bytes, 128U);
  TK_REQUIRE_EQ(lowered.arena_plan().workspace_bytes(), 128U);
  const ArenaAllocation* final_allocation =
      lowered.arena_plan().allocation_at(compute_count - 1U);
  TK_REQUIRE(final_allocation != nullptr);
  TK_REQUIRE_EQ(final_allocation->offset_bytes(), 64U);
  TK_REQUIRE(lowered.arena_plan().allocation_at(compute_count) == nullptr);
}

TK_TEST("Graph arena lowering rejects then accepts one beyond the default") {
  constexpr std::uint32_t compute_count =
      tensorkiln::kDefaultMaxArenaBuffers + 1U;
  const ScaleChainFixture fixture = make_scale_chain(compute_count);

  const auto default_result = GraphArenaLowering::run(
      fixture.graph,
      ArenaLimits{tensorkiln::kDefaultMaxArenaBuffers, 0U});
  TK_REQUIRE_EQ(
      require_error(default_result,
                    ErrorCode::arena_buffer_limit_exceeded)
          .message,
      "arena has 4097 buffer requests; limit is 4096");

  const ArenaLimits raised_limits{compute_count, 128U};
  const GraphArenaLoweringResult raised =
      unwrap(GraphArenaLowering::run(fixture.graph, raised_limits));
  TK_REQUIRE_EQ(raised.source_node_count(), compute_count + 1U);
  TK_REQUIRE_EQ(raised.execution_step_count(), compute_count);
  TK_REQUIRE_EQ(raised.requests().size(), compute_count);
  TK_REQUIRE_EQ(raised.arena_plan().limits(), raised_limits);
  TK_REQUIRE_EQ(raised.buffer_ordinal(fixture.result), compute_count - 1U);
  TK_REQUIRE_EQ(raised.requests()[compute_count - 2U],
                (ArenaBufferRequest{4U, compute_count - 2U,
                                    compute_count}));
  TK_REQUIRE_EQ(raised.requests()[compute_count - 1U],
                (ArenaBufferRequest{4U, compute_count - 1U,
                                    compute_count}));
  TK_REQUIRE_EQ(raised.arena_plan().stats().total_payload_bytes, 16388U);
  TK_REQUIRE_EQ(raised.arena_plan().stats().total_reserved_bytes, 262208U);
  TK_REQUIRE_EQ(raised.arena_plan().stats().peak_live_reserved_bytes, 128U);
  TK_REQUIRE_EQ(raised.arena_plan().workspace_bytes(), 128U);
  const ArenaAllocation* final_allocation =
      raised.arena_plan().allocation_at(compute_count - 1U);
  TK_REQUIRE(final_allocation != nullptr);
  TK_REQUIRE_EQ(final_allocation->offset_bytes(), 0U);
}

}  // namespace
