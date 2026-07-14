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
#include "tensorkiln/reference.hpp"

namespace {

using tensorkiln::ConstantOp;
using tensorkiln::DeadCodeElimination;
using tensorkiln::DeadCodeEliminationResult;
using tensorkiln::DeadCodeEliminationStats;
using tensorkiln::ElementType;
using tensorkiln::ErrorCode;
using tensorkiln::GraphBuilder;
using tensorkiln::GraphLimits;
using tensorkiln::InputBinding;
using tensorkiln::InputOp;
using tensorkiln::NodeProvenance;
using tensorkiln::OutputId;
using tensorkiln::ReferenceInterpreter;
using tensorkiln::ReferenceResult;
using tensorkiln::Shape;
using tensorkiln::ShapeLimits;
using tensorkiln::Tensor;
using tensorkiln::TensorLimits;
using tensorkiln::TensorType;
using tensorkiln::ValueId;
using tensorkiln::VerifiedGraph;

inline constexpr std::array<std::size_t, 7U> kRetainedSourceOrdinals{{
    0U, 1U, 2U, 4U, 5U, 7U, 9U,
}};
inline constexpr std::array<std::size_t, 3U> kDeadSourceOrdinals{{
    3U, 6U, 8U,
}};

[[nodiscard]] TensorType make_type(
    const std::initializer_list<std::int64_t> extents) {
  const auto shape = Shape::create(extents);
  TK_REQUIRE(shape.value_if() != nullptr);
  const auto type = TensorType::create(*shape.value_if());
  TK_REQUIRE(type.value_if() != nullptr);
  return *type.value_if();
}

[[nodiscard]] TensorType make_type(
    const std::initializer_list<std::int64_t> extents,
    const ShapeLimits shape_limits, const TensorLimits tensor_limits) {
  const auto shape = Shape::create(extents, shape_limits);
  TK_REQUIRE(shape.value_if() != nullptr);
  const auto type = TensorType::create(
      *shape.value_if(), ElementType::f32, tensor_limits);
  TK_REQUIRE(type.value_if() != nullptr);
  return *type.value_if();
}

[[nodiscard]] ValueId require_value(
    const tensorkiln::Result<ValueId>& result) {
  TK_REQUIRE(result.value_if() != nullptr);
  return *result.value_if();
}

void require_output(const tensorkiln::Result<OutputId>& result) {
  TK_REQUIRE(result.value_if() != nullptr);
}

[[nodiscard]] VerifiedGraph require_graph(
    tensorkiln::Result<VerifiedGraph> result) {
  TK_REQUIRE(result.value_if() != nullptr);
  return std::move(*result.value_if());
}

[[nodiscard]] DeadCodeEliminationResult require_dce(
    tensorkiln::Result<DeadCodeEliminationResult> result) {
  TK_REQUIRE(result.value_if() != nullptr);
  return std::move(*result.value_if());
}

[[nodiscard]] ReferenceResult require_reference(
    tensorkiln::Result<ReferenceResult> result) {
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

[[nodiscard]] const Tensor& require_output_tensor(
    const ReferenceResult& result, const std::string_view name) {
  const Tensor* tensor = result.output(name);
  TK_REQUIRE(tensor != nullptr);
  return *tensor;
}

void require_bits_equal(const std::span<const float> left,
                        const std::span<const float> right) {
  TK_REQUIRE_EQ(left.size(), right.size());
  for (std::size_t index = 0U; index < left.size(); ++index) {
    TK_REQUIRE_EQ(std::bit_cast<std::uint32_t>(left[index]),
                  std::bit_cast<std::uint32_t>(right[index]));
  }
}

struct InterleavedFixture final {
  VerifiedGraph graph;
  std::array<ValueId, 10U> values;
};

class DeterministicGenerator final {
 public:
  explicit DeterministicGenerator(const std::uint32_t seed) noexcept
      : state_(seed) {}

  [[nodiscard]] std::uint32_t next() noexcept {
    state_ = state_ * UINT32_C(1664525) + UINT32_C(1013904223);
    return state_;
  }

  [[nodiscard]] std::size_t index(const std::size_t bound) noexcept {
    return static_cast<std::size_t>(
        next() % static_cast<std::uint32_t>(bound));
  }

 private:
  std::uint32_t state_;
};

[[nodiscard]] InterleavedFixture build_interleaved_fixture() {
  GraphBuilder builder;
  const ValueId input =
      require_value(builder.input("x", make_type({2, 3})));
  const ValueId unused_input =
      require_value(builder.input("unused", make_type({2})));

  const std::array<float, 6U> weights_data{{
      1.0F, -2.0F,
      0.5F, 3.0F,
      -1.0F, 4.0F,
  }};
  const ValueId weights = require_value(
      builder.constant("weights", make_type({3, 2}), weights_data));

  const std::array<float, 2U> dead_data{{10.0F, -10.0F}};
  const ValueId dead_constant = require_value(
      builder.constant("dead_bias", make_type({2}), dead_data));
  const ValueId product = require_value(builder.matmul(input, weights));

  const std::array<float, 2U> bias_data{{-0.25F, 0.5F}};
  const ValueId bias = require_value(
      builder.constant("bias", make_type({2}), bias_data));
  const ValueId dead_add =
      require_value(builder.add(unused_input, dead_constant));
  const ValueId pre_activation = require_value(builder.add(product, bias));
  const ValueId dead_relu = require_value(builder.relu(dead_add));
  const ValueId result = require_value(builder.relu(pre_activation));

  require_output(builder.output("result", result));
  require_output(builder.output("pre_a", pre_activation));
  require_output(builder.output("pre_b", pre_activation));

  return InterleavedFixture{
      require_graph(std::move(builder).finish()),
      {input, unused_input, weights, dead_constant, product, bias, dead_add,
       pre_activation, dead_relu, result},
  };
}

TK_TEST("DCE removes interleaved dead work without changing the feed or output ABI") {
  const InterleavedFixture fixture = build_interleaved_fixture();
  const DeadCodeEliminationResult eliminated =
      require_dce(DeadCodeElimination::run(fixture.graph));

  const DeadCodeEliminationStats expected_stats{
      10U, 7U, 3U, 10U, 8U, 2U,
  };
  TK_REQUIRE_EQ(eliminated.stats(), expected_stats);
  TK_REQUIRE_EQ(eliminated.graph().limits(), fixture.graph.limits());
  TK_REQUIRE_EQ(eliminated.graph().nodes().size(), 7U);
  TK_REQUIRE_EQ(eliminated.graph().outputs().size(), 3U);

  const std::span<const tensorkiln::Node> nodes = eliminated.graph().nodes();
  TK_REQUIRE(std::holds_alternative<InputOp>(nodes[0].operation()));
  TK_REQUIRE(std::holds_alternative<InputOp>(nodes[1].operation()));
  TK_REQUIRE_EQ(std::get<InputOp>(nodes[0].operation()).name, "x");
  TK_REQUIRE_EQ(std::get<InputOp>(nodes[1].operation()).name, "unused");
  TK_REQUIRE(std::holds_alternative<ConstantOp>(nodes[2].operation()));
  TK_REQUIRE(std::holds_alternative<tensorkiln::MatMulOp>(
      nodes[3].operation()));
  TK_REQUIRE(std::holds_alternative<ConstantOp>(nodes[4].operation()));
  TK_REQUIRE(std::holds_alternative<tensorkiln::AddOp>(
      nodes[5].operation()));
  TK_REQUIRE(std::holds_alternative<tensorkiln::ReluOp>(
      nodes[6].operation()));
  TK_REQUIRE_EQ(nodes[3].inputs()[0].ordinal(), 0U);
  TK_REQUIRE_EQ(nodes[3].inputs()[1].ordinal(), 2U);
  TK_REQUIRE_EQ(nodes[5].inputs()[0].ordinal(), 3U);
  TK_REQUIRE_EQ(nodes[5].inputs()[1].ordinal(), 4U);
  TK_REQUIRE_EQ(nodes[6].inputs()[0].ordinal(), 5U);

  const std::span<const tensorkiln::GraphOutput> outputs =
      eliminated.graph().outputs();
  TK_REQUIRE_EQ(outputs[0].id().ordinal, 0U);
  TK_REQUIRE_EQ(outputs[0].name(), "result");
  TK_REQUIRE_EQ(outputs[0].value().ordinal(), 6U);
  TK_REQUIRE_EQ(outputs[1].id().ordinal, 1U);
  TK_REQUIRE_EQ(outputs[1].name(), "pre_a");
  TK_REQUIRE_EQ(outputs[1].value().ordinal(), 5U);
  TK_REQUIRE_EQ(outputs[2].id().ordinal, 2U);
  TK_REQUIRE_EQ(outputs[2].name(), "pre_b");
  TK_REQUIRE_EQ(outputs[2].value(), outputs[1].value());

  const std::string expected_dump =
      "tensorkiln.dce v0 {\n"
      "  nodes {source=10, retained=7, removed=3}\n"
      "  constant_elements {source=10, retained=8, removed=2}\n"
      "  #n0 %0 <- {source #n0 %0}\n"
      "  #n1 %1 <- {source #n1 %1}\n"
      "  #n2 %2 <- {source #n2 %2}\n"
      "  #n3 %3 <- {source #n4 %4}\n"
      "  #n4 %4 <- {source #n5 %5}\n"
      "  #n5 %5 <- {source #n7 %7}\n"
      "  #n6 %6 <- {source #n9 %9}\n"
      "}\n";
  TK_REQUIRE_EQ(eliminated.dump(), expected_dump);

  const std::string expected_provenance_dump =
      "tensorkiln.provenance v0 {\n"
      "  #n0 %0 <- {source #n0 %0}\n"
      "  #n1 %1 <- {source #n1 %1}\n"
      "  #n2 %2 <- {source #n2 %2}\n"
      "  #n3 %3 <- {source #n4 %4}\n"
      "  #n4 %4 <- {source #n5 %5}\n"
      "  #n5 %5 <- {source #n7 %7}\n"
      "  #n6 %6 <- {source #n9 %9}\n"
      "}\n";
  TK_REQUIRE_EQ(eliminated.provenance().dump(), expected_provenance_dump);
}

TK_TEST("DCE is bitwise reference equivalent while removing dead resource work") {
  const InterleavedFixture fixture = build_interleaved_fixture();
  const DeadCodeEliminationResult eliminated =
      require_dce(DeadCodeElimination::run(fixture.graph));

  const std::array<float, 6U> input_data{{
      1.0F, 2.0F, 3.0F,
      -1.0F, 0.5F, 2.0F,
  }};
  const std::array<float, 2U> unused_data{{
      -0.0F, std::bit_cast<float>(UINT32_C(0x7fc12345)),
  }};
  const std::array<InputBinding, 2U> bindings{{
      InputBinding{"x", input_data},
      InputBinding{"unused", unused_data},
  }};

  const ReferenceResult source_result = require_reference(
      ReferenceInterpreter::run(fixture.graph, bindings));
  const ReferenceResult result = require_reference(
      ReferenceInterpreter::run(eliminated.graph(), bindings));

  constexpr std::array<std::string_view, 3U> output_names{{
      "result", "pre_a", "pre_b",
  }};
  for (const std::string_view name : output_names) {
    const Tensor& source_tensor = require_output_tensor(source_result, name);
    const Tensor& result_tensor = require_output_tensor(result, name);
    TK_REQUIRE_EQ(result_tensor.type(), source_tensor.type());
    require_bits_equal(result_tensor.data(), source_tensor.data());
  }

  TK_REQUIRE(source_result.output("pre_a") == source_result.output("pre_b"));
  TK_REQUIRE(result.output("pre_a") == result.output("pre_b"));
  TK_REQUIRE(source_result.value(fixture.values[3]) != nullptr);
  TK_REQUIRE(source_result.value(fixture.values[6]) != nullptr);
  TK_REQUIRE(source_result.value(fixture.values[8]) != nullptr);
  TK_REQUIRE(result.value(fixture.values[9]) == nullptr);

  TK_REQUIRE_EQ(source_result.materialized_bytes(), 136U);
  TK_REQUIRE_EQ(result.materialized_bytes(), 112U);
  TK_REQUIRE_EQ(source_result.scalar_steps(), 42U);
  TK_REQUIRE_EQ(result.scalar_steps(), 36U);
}

TK_TEST("DCE copies live constant bits and fingerprints without canonicalization") {
  GraphBuilder builder;
  const std::array<float, 5U> live_data{{
      0.0F,
      -0.0F,
      std::bit_cast<float>(UINT32_C(0x7fc12345)),
      std::bit_cast<float>(UINT32_C(0x80000001)),
      std::bit_cast<float>(UINT32_C(0x7f812345)),
  }};
  TK_REQUIRE_EQ(std::bit_cast<std::uint32_t>(live_data[4]), 0x7f812345U);
  const std::array<float, 4U> dead_data{{1.0F, 2.0F, 3.0F, 4.0F}};
  const ValueId live = require_value(
      builder.constant("live", make_type({5}), live_data));
  static_cast<void>(
      require_value(builder.constant("dead", make_type({4}), dead_data)));
  require_output(builder.output("result", live));
  const VerifiedGraph source = require_graph(std::move(builder).finish());
  VerifiedGraph copied_source = source;
  copied_source = source;
  const ConstantOp& copied_constant =
      std::get<ConstantOp>(copied_source.nodes()[0].operation());
  require_bits_equal(copied_constant.data, live_data);

  const DeadCodeEliminationResult eliminated =
      require_dce(DeadCodeElimination::run(source));
  TK_REQUIRE_EQ(eliminated.graph().nodes().size(), 1U);
  const ConstantOp& source_constant =
      std::get<ConstantOp>(source.nodes()[0].operation());
  const ConstantOp& result_constant =
      std::get<ConstantOp>(eliminated.graph().nodes()[0].operation());
  require_bits_equal(source_constant.data, live_data);
  TK_REQUIRE_EQ(result_constant.name, source_constant.name);
  TK_REQUIRE_EQ(result_constant.fingerprint, source_constant.fingerprint);
  require_bits_equal(result_constant.data, source_constant.data);

  const DeadCodeEliminationStats expected_stats{
      2U, 1U, 1U, 9U, 5U, 4U,
  };
  TK_REQUIRE_EQ(eliminated.stats(), expected_stats);

  const ReferenceResult source_result = require_reference(
      ReferenceInterpreter::run(source, {}));
  const ReferenceResult result = require_reference(
      ReferenceInterpreter::run(eliminated.graph(), {}));
  require_bits_equal(require_output_tensor(result, "result").data(),
                     require_output_tensor(source_result, "result").data());
  require_bits_equal(require_output_tensor(result, "result").data(),
                     std::span<const float>{live_data});
}

TK_TEST("DCE provenance uses complete owner-tagged handles in both directions") {
  const InterleavedFixture fixture = build_interleaved_fixture();
  const DeadCodeEliminationResult eliminated =
      require_dce(DeadCodeElimination::run(fixture.graph));

  TK_REQUIRE_EQ(eliminated.provenance().entries().size(),
                kRetainedSourceOrdinals.size());
  for (std::size_t result_index = 0U;
       result_index < kRetainedSourceOrdinals.size(); ++result_index) {
    const std::size_t source_index = kRetainedSourceOrdinals[result_index];
    const tensorkiln::Node& result_node =
        eliminated.graph().nodes()[result_index];
    const tensorkiln::Node& source_node = fixture.graph.nodes()[source_index];

    const NodeProvenance* by_result_node =
        eliminated.provenance().for_result(result_node.id());
    const NodeProvenance* by_result_value =
        eliminated.provenance().for_result(result_node.output());
    TK_REQUIRE(by_result_node != nullptr);
    TK_REQUIRE(by_result_node == by_result_value);
    TK_REQUIRE_EQ(by_result_node->sources().size(), 1U);
    TK_REQUIRE_EQ(by_result_node->sources()[0].node(), source_node.id());
    TK_REQUIRE_EQ(by_result_node->sources()[0].value(), source_node.output());
    TK_REQUIRE(eliminated.provenance().for_source(source_node.id()) ==
               by_result_node);
    TK_REQUIRE(eliminated.provenance().for_source(source_node.output()) ==
               by_result_node);
  }

  for (const std::size_t source_index : kDeadSourceOrdinals) {
    const tensorkiln::Node& dead = fixture.graph.nodes()[source_index];
    TK_REQUIRE(eliminated.provenance().for_source(dead.id()) == nullptr);
    TK_REQUIRE(eliminated.provenance().for_source(dead.output()) == nullptr);
  }

  TK_REQUIRE(eliminated.provenance().for_result(
                 fixture.graph.nodes()[0].id()) == nullptr);
  TK_REQUIRE(eliminated.provenance().for_result(fixture.values[0]) == nullptr);
  TK_REQUIRE(eliminated.provenance().for_source(
                 eliminated.graph().nodes()[0].id()) == nullptr);
  TK_REQUIRE(eliminated.provenance().for_source(
                 eliminated.graph().nodes()[0].output()) == nullptr);
  TK_REQUIRE(fixture.graph.node(eliminated.graph().nodes()[0].id()) == nullptr);
  TK_REQUIRE(fixture.graph.type(eliminated.graph().nodes()[0].output()) ==
             nullptr);
  TK_REQUIRE(eliminated.graph().node(fixture.graph.nodes()[0].id()) == nullptr);
  TK_REQUIRE(eliminated.graph().type(fixture.values[0]) == nullptr);

  GraphBuilder foreign_builder;
  const ValueId foreign_value =
      require_value(foreign_builder.input("x", make_type({2, 3})));
  require_output(foreign_builder.output("result", foreign_value));
  const VerifiedGraph foreign =
      require_graph(std::move(foreign_builder).finish());
  TK_REQUIRE_EQ(foreign.nodes()[0].id().ordinal(), 0U);
  TK_REQUIRE(eliminated.provenance().for_result(foreign.nodes()[0].id()) ==
             nullptr);
  TK_REQUIRE(eliminated.provenance().for_result(foreign_value) == nullptr);
  TK_REQUIRE(eliminated.provenance().for_source(foreign.nodes()[0].id()) ==
             nullptr);
  TK_REQUIRE(eliminated.provenance().for_source(foreign_value) == nullptr);
}

TK_TEST("DCE reuses exact non-default graph shape tensor and name limits") {
  const std::uint64_t element_count =
      tensorkiln::kDefaultMaxElements + UINT64_C(1);
  const ShapeLimits shape_limits{element_count};
  const TensorLimits tensor_limits{element_count * UINT64_C(4)};
  const GraphLimits limits{
      2U,
      1U,
      192U,
      0U,
      shape_limits,
      tensor_limits,
  };
  const std::string input_name = "input_" + std::string(140U, 'x');
  const std::string output_name = "result_" + std::string(140U, 'y');
  const auto extent = static_cast<std::int64_t>(element_count);

  GraphBuilder builder(limits);
  const ValueId input = require_value(builder.input(
      input_name, make_type({extent}, shape_limits, tensor_limits)));
  require_output(builder.output(output_name, input));
  const VerifiedGraph source = require_graph(std::move(builder).finish());

  const DeadCodeEliminationResult eliminated =
      require_dce(DeadCodeElimination::run(source));
  TK_REQUIRE_EQ(eliminated.graph().limits(), limits);
  TK_REQUIRE_EQ(eliminated.graph().dump(), source.dump());
  TK_REQUIRE_EQ(eliminated.graph().nodes()[0].output_type().numel(),
                element_count);
  TK_REQUIRE_EQ(std::get<InputOp>(
                    eliminated.graph().nodes()[0].operation()).name,
                input_name);
  TK_REQUIRE_EQ(eliminated.graph().outputs()[0].name(), output_name);
  const DeadCodeEliminationStats expected_stats{
      1U, 1U, 0U, 0U, 0U, 0U,
  };
  TK_REQUIRE_EQ(eliminated.stats(), expected_stats);
}

TK_TEST("DCE is idempotent and composes lineage back to the original graph") {
  const InterleavedFixture fixture = build_interleaved_fixture();
  const DeadCodeEliminationResult first =
      require_dce(DeadCodeElimination::run(fixture.graph));
  const std::string first_graph_dump = first.graph().dump();

  const DeadCodeEliminationResult second = require_dce(
      DeadCodeElimination::run(first.graph(), first.provenance()));
  const DeadCodeEliminationStats expected_stats{
      7U, 7U, 0U, 8U, 8U, 0U,
  };
  TK_REQUIRE_EQ(second.stats(), expected_stats);
  TK_REQUIRE_EQ(second.graph().dump(), first_graph_dump);
  TK_REQUIRE_EQ(second.provenance().dump(), first.provenance().dump());

  for (std::size_t result_index = 0U;
       result_index < kRetainedSourceOrdinals.size(); ++result_index) {
    const std::size_t source_index = kRetainedSourceOrdinals[result_index];
    const NodeProvenance* entry = second.provenance().for_result(
        second.graph().nodes()[result_index].id());
    TK_REQUIRE(entry != nullptr);
    TK_REQUIRE_EQ(entry->sources().size(), 1U);
    TK_REQUIRE_EQ(entry->sources()[0].node(),
                  fixture.graph.nodes()[source_index].id());
    TK_REQUIRE_EQ(entry->sources()[0].value(),
                  fixture.graph.nodes()[source_index].output());
  }
}

TK_TEST("DCE is deterministic and bit-exact across seeded scalar DAGs") {
  constexpr std::uint32_t kCaseCount = 48U;
  constexpr std::size_t kGeneratedDefinitions = 24U;

  for (std::uint32_t case_index = 0U; case_index < kCaseCount;
       ++case_index) {
    DeterministicGenerator generator(
        UINT32_C(0x5eed1234) ^ (case_index * UINT32_C(0x9e3779b9)));
    GraphBuilder builder;
    const TensorType scalar_type = make_type({});
    const ValueId x = require_value(builder.input("x", scalar_type));
    const ValueId y = require_value(builder.input("y", scalar_type));
    std::vector<ValueId> values{x, y};
    values.reserve(2U + kGeneratedDefinitions);

    for (std::size_t step = 0U; step < kGeneratedDefinitions; ++step) {
      const std::uint32_t selector = generator.next() % 4U;
      ValueId generated = x;
      if (selector == 0U) {
        const std::int32_t numerator =
            static_cast<std::int32_t>(generator.next() % 2049U) - 1024;
        const std::array<float, 1U> data{{
            static_cast<float>(numerator) / 64.0F,
        }};
        generated = require_value(builder.constant(
            "c_" + std::to_string(step), scalar_type, data));
      } else if (selector == 1U) {
        generated =
            require_value(builder.relu(values[generator.index(values.size())]));
      } else {
        const ValueId left = values[generator.index(values.size())];
        const ValueId right = (generator.next() & 3U) == 0U
                                  ? left
                                  : values[generator.index(values.size())];
        generated = require_value(builder.add(left, right));
      }
      values.push_back(generated);
    }

    const ValueId result = values[2U + generator.index(14U)];
    require_output(builder.output("result", result));
    require_output(builder.output("alias", result));
    const VerifiedGraph source = require_graph(std::move(builder).finish());

    const DeadCodeEliminationResult first =
        require_dce(DeadCodeElimination::run(source));
    const DeadCodeEliminationResult second =
        require_dce(DeadCodeElimination::run(source));
    TK_REQUIRE(first.stats().removed_nodes >= 10U);
    TK_REQUIRE_EQ(first.stats(), second.stats());
    TK_REQUIRE_EQ(first.graph().dump(), second.graph().dump());
    TK_REQUIRE_EQ(first.provenance().dump(), second.provenance().dump());
    TK_REQUIRE_EQ(first.dump(), second.dump());

    const std::array<float, 1U> x_data{{
        static_cast<float>(case_index) / 16.0F - 1.25F,
    }};
    const std::array<float, 1U> y_data{{
        0.75F - static_cast<float>(case_index) / 32.0F,
    }};
    const std::array<InputBinding, 2U> bindings{{
        InputBinding{"x", x_data},
        InputBinding{"y", y_data},
    }};
    const ReferenceResult source_result =
        require_reference(ReferenceInterpreter::run(source, bindings));
    const ReferenceResult optimized_result = require_reference(
        ReferenceInterpreter::run(first.graph(), bindings));

    const Tensor& source_output =
        require_output_tensor(source_result, "result");
    const Tensor& optimized_output =
        require_output_tensor(optimized_result, "result");
    TK_REQUIRE_EQ(optimized_output.type(), source_output.type());
    require_bits_equal(optimized_output.data(), source_output.data());
    TK_REQUIRE(source_result.output("result") == source_result.output("alias"));
    TK_REQUIRE(optimized_result.output("result") ==
               optimized_result.output("alias"));
    TK_REQUIRE(optimized_result.materialized_bytes() <
               source_result.materialized_bytes());
    TK_REQUIRE(optimized_result.scalar_steps() < source_result.scalar_steps());
  }
}

TK_TEST("DCE rejects incomplete and foreign provenance with stable diagnostics") {
  const InterleavedFixture fixture = build_interleaved_fixture();
  const DeadCodeEliminationResult first =
      require_dce(DeadCodeElimination::run(fixture.graph));

  const auto incomplete =
      DeadCodeElimination::run(fixture.graph, first.provenance());
  const tensorkiln::Diagnostic& incomplete_error =
      require_error(incomplete, ErrorCode::provenance_domain_mismatch);
  TK_REQUIRE_EQ(incomplete_error.message,
                "source provenance has 7 entries for a graph with 10 nodes");

  const InterleavedFixture foreign_fixture = build_interleaved_fixture();
  const DeadCodeEliminationResult foreign =
      require_dce(DeadCodeElimination::run(foreign_fixture.graph));
  const auto wrong_owner =
      DeadCodeElimination::run(first.graph(), foreign.provenance());
  const tensorkiln::Diagnostic& owner_error =
      require_error(wrong_owner, ErrorCode::provenance_domain_mismatch);
  TK_REQUIRE_EQ(owner_error.message,
                "source provenance does not describe #n0 %0");

  const auto wrong_composition =
      first.provenance().compose(foreign.provenance());
  const tensorkiln::Diagnostic& composition_error =
      require_error(wrong_composition, ErrorCode::provenance_domain_mismatch);
  TK_REQUIRE_EQ(composition_error.message,
                "upstream provenance does not describe #n0 %0");

  TK_REQUIRE_EQ(tensorkiln::error_code_name(
                    ErrorCode::compiler_internal_invariant),
                "compiler_internal_invariant");
  TK_REQUIRE_EQ(tensorkiln::error_code_name(
                    ErrorCode::provenance_domain_mismatch),
                "provenance_domain_mismatch");
}

}  // namespace
