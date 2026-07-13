#include "test.hpp"

#include <bit>
#include <cstdint>
#include <initializer_list>
#include <span>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "tensorkiln/graph.hpp"

namespace {

using tensorkiln::ConstantOp;
using tensorkiln::ElementType;
using tensorkiln::ErrorCode;
using tensorkiln::GraphBuilder;
using tensorkiln::GraphLimits;
using tensorkiln::Node;
using tensorkiln::NodeId;
using tensorkiln::OutputId;
using tensorkiln::Shape;
using tensorkiln::ShapeLimits;
using tensorkiln::TensorLimits;
using tensorkiln::TensorType;
using tensorkiln::ValueId;
using tensorkiln::VerifiedGraph;

[[nodiscard]] TensorType make_type(
    const std::initializer_list<std::int64_t> extents) {
  const auto shape = Shape::create(extents);
  TK_REQUIRE(shape.value_if() != nullptr);
  const auto type = TensorType::create(*shape.value_if());
  TK_REQUIRE(type.value_if() != nullptr);
  return *type.value_if();
}

ValueId require_value(
    const tensorkiln::Result<ValueId>& result) {
  TK_REQUIRE(result.value_if() != nullptr);
  return *result.value_if();
}

OutputId require_output(
    const tensorkiln::Result<OutputId>& result) {
  TK_REQUIRE(result.value_if() != nullptr);
  return *result.value_if();
}

[[nodiscard]] VerifiedGraph require_graph(
    tensorkiln::Result<VerifiedGraph> result) {
  TK_REQUIRE(result.value_if() != nullptr);
  return std::move(*result.value_if());
}

[[nodiscard]] const Node& require_node(const VerifiedGraph& graph,
                                       const NodeId id,
                                       const std::size_t arity) {
  const Node* node = graph.node(id);
  TK_REQUIRE(node != nullptr);
  TK_REQUIRE_EQ(node->inputs().size(), arity);
  return *node;
}

[[nodiscard]] const TensorType& require_type(const VerifiedGraph& graph,
                                             const ValueId value) {
  const TensorType* type = graph.type(value);
  TK_REQUIRE(type != nullptr);
  return *type;
}

template <typename T>
const tensorkiln::Diagnostic& require_error(
    const tensorkiln::Result<T>& result, const ErrorCode code) {
  TK_REQUIRE(result.error_if() != nullptr);
  TK_REQUIRE_EQ(result.error_if()->code, code);
  return *result.error_if();
}

[[nodiscard]] VerifiedGraph build_reference_graph() {
  GraphBuilder builder;
  const ValueId input = require_value(builder.input("x", make_type({2, 3})));
  const std::vector<float> bias_data{1.0F, -0.0F, 2.0F};
  const ValueId bias = require_value(
      builder.constant("bias", make_type({3}), bias_data));
  const ValueId sum = require_value(builder.add(input, bias));
  const ValueId activated = require_value(builder.relu(sum));
  require_output(builder.output("result", activated));
  return require_graph(std::move(builder).finish());
}

TK_TEST("Graph IDs are dense stable and topological") {
  GraphBuilder builder;
  const ValueId input = require_value(builder.input("x", make_type({2, 3})));
  const std::vector<float> bias_data{1.0F, 2.0F, 3.0F};
  const ValueId bias = require_value(
      builder.constant("bias", make_type({3}), bias_data));
  const ValueId sum = require_value(builder.add(input, bias));
  const ValueId activated = require_value(builder.relu(sum));
  const OutputId first = require_output(builder.output("sum", sum));
  const OutputId second = require_output(builder.output("result", activated));
  const VerifiedGraph graph = require_graph(std::move(builder).finish());

  TK_REQUIRE_EQ(input.ordinal(), 0U);
  TK_REQUIRE_EQ(bias.ordinal(), 1U);
  TK_REQUIRE_EQ(sum.ordinal(), 2U);
  TK_REQUIRE_EQ(activated.ordinal(), 3U);
  TK_REQUIRE_EQ(first.ordinal, 0U);
  TK_REQUIRE_EQ(second.ordinal, 1U);
  TK_REQUIRE_EQ(graph.nodes().size(), 4U);
  TK_REQUIRE_EQ(graph.outputs().size(), 2U);
  const Node& add = require_node(graph, NodeId{2U}, 2U);
  TK_REQUIRE_EQ(add.inputs()[0], input);
  TK_REQUIRE_EQ(add.inputs()[1], bias);
  TK_REQUIRE_EQ(add.output_type().to_string(), "f32[2,3]");
}

TK_TEST("Failed Add is fully transactional") {
  GraphBuilder candidate;
  const ValueId left =
      require_value(candidate.input("left", make_type({2, 3})));
  const ValueId invalid =
      require_value(candidate.input("invalid", make_type({4, 5})));
  const ValueId bias =
      require_value(candidate.input("bias", make_type({3})));

  require_error(candidate.add(left, invalid), ErrorCode::broadcast_incompatible);
  TK_REQUIRE_EQ(candidate.node_count(), 3U);
  const ValueId sum = require_value(candidate.add(left, bias));
  TK_REQUIRE_EQ(sum.ordinal(), 3U);
  require_output(candidate.output("result", sum));
  const VerifiedGraph candidate_graph =
      require_graph(std::move(candidate).finish());

  GraphBuilder control;
  const ValueId control_left =
      require_value(control.input("left", make_type({2, 3})));
  require_value(control.input("invalid", make_type({4, 5})));
  const ValueId control_bias =
      require_value(control.input("bias", make_type({3})));
  const ValueId control_sum =
      require_value(control.add(control_left, control_bias));
  require_output(control.output("result", control_sum));
  const VerifiedGraph control_graph = require_graph(std::move(control).finish());

  TK_REQUIRE_EQ(candidate_graph.dump(), control_graph.dump());
}

TK_TEST("Failed constant does not reserve its name or ID") {
  GraphBuilder builder;
  const std::vector<float> short_data{1.0F, 2.0F};
  require_error(builder.constant("bias", make_type({3}), short_data),
                ErrorCode::constant_size_mismatch);
  TK_REQUIRE_EQ(builder.node_count(), 0U);
  TK_REQUIRE_EQ(builder.constant_elements(), 0U);

  const std::vector<float> exact_data{1.0F, 2.0F, 3.0F};
  const ValueId bias = require_value(
      builder.constant("bias", make_type({3}), exact_data));
  TK_REQUIRE_EQ(bias.ordinal(), 0U);
}

TK_TEST("Graph rejects values owned by another builder") {
  GraphBuilder first;
  GraphBuilder second;
  const ValueId foreign = require_value(first.input("x", make_type({2, 3})));
  const ValueId local = require_value(second.input("x", make_type({2, 3})));

  require_error(second.relu(foreign), ErrorCode::value_not_found);
  require_error(second.add(local, foreign), ErrorCode::value_not_found);
  require_error(second.output("bad", foreign), ErrorCode::value_not_found);
  TK_REQUIRE_EQ(second.node_count(), 1U);
  TK_REQUIRE_EQ(second.output_count(), 0U);
}

TK_TEST("Definition names use a narrow deterministic grammar") {
  GraphBuilder builder(GraphLimits{4096U, 64U, 5U});

  require_error(builder.input("", make_type({1})), ErrorCode::invalid_name);
  require_error(builder.input("9bad", make_type({1})),
                ErrorCode::invalid_name);
  require_error(builder.input("a/b", make_type({1})), ErrorCode::invalid_name);
  require_error(builder.input("abcdef", make_type({1})),
                ErrorCode::name_limit_exceeded);
  const ValueId value = require_value(builder.input("valid", make_type({1})));
  require_error(builder.input("valid", make_type({1})),
                ErrorCode::duplicate_name);
  TK_REQUIRE_EQ(value.ordinal(), 0U);

  GraphBuilder accepted(GraphLimits{4096U, 64U, 5U});
  require_value(accepted.input("_a.1", make_type({1})));
}

TK_TEST("Constant owns caller data and distinguishes signed zero") {
  GraphBuilder builder;
  std::vector<float> source{0.0F, -0.0F};
  const ValueId constant = require_value(
      builder.constant("zeros", make_type({2}), source));
  source[0] = 9.0F;
  source.clear();
  require_output(builder.output("result", constant));
  const VerifiedGraph graph = require_graph(std::move(builder).finish());

  const Node& node = require_node(graph, NodeId{0U}, 0U);
  const auto* operation = std::get_if<ConstantOp>(&node.operation());
  TK_REQUIRE(operation != nullptr);
  TK_REQUIRE_EQ(operation->data.size(), 2U);
  TK_REQUIRE_EQ(std::bit_cast<std::uint32_t>(operation->data[0]), 0x00000000U);
  TK_REQUIRE_EQ(std::bit_cast<std::uint32_t>(operation->data[1]), 0x80000000U);

  GraphBuilder positive_builder;
  const std::vector<float> positive_zeros{0.0F, 0.0F};
  const ValueId positive = require_value(positive_builder.constant(
      "zeros", make_type({2}), positive_zeros));
  require_output(positive_builder.output("result", positive));
  const VerifiedGraph positive_graph =
      require_graph(std::move(positive_builder).finish());
  const Node& positive_node = require_node(positive_graph, NodeId{0U}, 0U);
  const auto* positive_operation =
      std::get_if<ConstantOp>(&positive_node.operation());
  TK_REQUIRE(positive_operation != nullptr);
  TK_REQUIRE(operation->fingerprint != positive_operation->fingerprint);
}

TK_TEST("Add and Relu infer their output types") {
  GraphBuilder builder;
  const ValueId tensor =
      require_value(builder.input("tensor", make_type({2, 1, 4})));
  const ValueId bias = require_value(builder.input("bias", make_type({3, 4})));
  const ValueId sum = require_value(builder.add(tensor, bias));
  const ValueId activated = require_value(builder.relu(sum));
  require_output(builder.output("result", activated));
  const VerifiedGraph graph = require_graph(std::move(builder).finish());

  TK_REQUIRE_EQ(require_type(graph, sum).to_string(), "f32[2,3,4]");
  TK_REQUIRE_EQ(require_type(graph, sum), require_type(graph, activated));
}

TK_TEST("Matmul infers a broadcast batch type") {
  GraphBuilder builder;
  const ValueId left =
      require_value(builder.input("left", make_type({2, 1, 3, 5})));
  const ValueId right =
      require_value(builder.input("right", make_type({1, 4, 5, 7})));
  const ValueId product = require_value(builder.matmul(left, right));
  require_output(builder.output("result", product));
  const VerifiedGraph graph = require_graph(std::move(builder).finish());

  TK_REQUIRE_EQ(require_type(graph, product).to_string(), "f32[2,4,3,7]");
}

TK_TEST("Output labels are unique but may alias definitions") {
  GraphBuilder builder;
  const ValueId input = require_value(builder.input("x", make_type({2})));
  const OutputId first = require_output(builder.output("x", input));
  require_error(builder.output("x", input), ErrorCode::duplicate_name);
  const OutputId second = require_output(builder.output("copy", input));

  TK_REQUIRE_EQ(first.ordinal, 0U);
  TK_REQUIRE_EQ(second.ordinal, 1U);
}

TK_TEST("Graph node limit is exact and transactional") {
  GraphBuilder builder(GraphLimits{2U});
  const ValueId input = require_value(builder.input("x", make_type({2})));
  const ValueId first = require_value(builder.relu(input));
  require_error(builder.relu(first), ErrorCode::graph_node_limit_exceeded);
  TK_REQUIRE_EQ(builder.node_count(), 2U);
}

TK_TEST("Zero graph node limit rejects the first definition") {
  GraphBuilder builder(GraphLimits{0U});

  require_error(builder.input("x", make_type({1})),
                ErrorCode::graph_node_limit_exceeded);
  TK_REQUIRE_EQ(builder.node_count(), 0U);
}

TK_TEST("Graph output limit is exact and transactional") {
  GraphBuilder builder(GraphLimits{4096U, 1U});
  const ValueId input = require_value(builder.input("x", make_type({2})));
  const OutputId first = require_output(builder.output("first", input));
  require_error(builder.output("second", input),
                ErrorCode::graph_output_limit_exceeded);
  TK_REQUIRE_EQ(first.ordinal, 0U);
  TK_REQUIRE_EQ(builder.output_count(), 1U);
}

TK_TEST("Zero graph output limit rejects the first label") {
  GraphBuilder builder(GraphLimits{4096U, 0U});
  const ValueId input = require_value(builder.input("x", make_type({1})));

  require_error(builder.output("result", input),
                ErrorCode::graph_output_limit_exceeded);
  TK_REQUIRE_EQ(builder.output_count(), 0U);
}

TK_TEST("Aggregate constant limit spans every constant") {
  GraphBuilder builder(GraphLimits{4096U, 64U, 128U, 4U});
  const std::vector<float> two{1.0F, 2.0F};
  const std::vector<float> one{3.0F};
  require_value(builder.constant("first", make_type({2}), two));
  require_value(builder.constant("second", make_type({1}), one));
  require_error(builder.constant("third", make_type({2}), two),
                ErrorCode::constant_element_limit_exceeded);
  const ValueId third =
      require_value(builder.constant("third", make_type({1}), one));

  TK_REQUIRE_EQ(third.ordinal(), 2U);
  TK_REQUIRE_EQ(builder.node_count(), 3U);
  TK_REQUIRE_EQ(builder.constant_elements(), 4U);
}

TK_TEST("Input types obey graph element and byte limits") {
  GraphLimits element_limits;
  element_limits.shape_limits = ShapeLimits{7U};
  GraphBuilder element_builder(element_limits);
  require_error(element_builder.input("too_large", make_type({8})),
                ErrorCode::element_limit_exceeded);
  const ValueId exact =
      require_value(element_builder.input("exact", make_type({7})));
  TK_REQUIRE_EQ(exact.ordinal(), 0U);

  GraphLimits byte_limits;
  byte_limits.tensor_limits = TensorLimits{7U};
  GraphBuilder byte_builder(byte_limits);
  require_error(byte_builder.input("too_wide", make_type({2})),
                ErrorCode::byte_limit_exceeded);
  TK_REQUIRE_EQ(byte_builder.node_count(), 0U);
}

TK_TEST("Derived types obey graph shape limits transactionally") {
  GraphLimits limits;
  limits.shape_limits = ShapeLimits{63U};
  limits.tensor_limits = TensorLimits{252U};
  GraphBuilder builder(limits);
  const ValueId left = require_value(builder.input("left", make_type({8, 1})));
  const ValueId right =
      require_value(builder.input("right", make_type({1, 8})));

  require_error(builder.add(left, right), ErrorCode::element_limit_exceeded);
  TK_REQUIRE_EQ(builder.node_count(), 2U);
}

TK_TEST("Canonical dump is deterministic and graph-owned") {
  const VerifiedGraph first = build_reference_graph();
  const VerifiedGraph second = build_reference_graph();

  TK_REQUIRE_EQ(first.dump(), second.dump());
  TK_REQUIRE_EQ(first.dump(),
                "tensorkiln.graph v0 {\n"
                "  #n0 %0 = input @x : f32[2,3]\n"
                "  #n1 %1 = constant @bias {elements=3, "
                "fnv1a64=0x99e02dab84d74dd8} : f32[3]\n"
                "  #n2 %2 = add %0, %1 : f32[2,3]\n"
                "  #n3 %3 = relu %2 : f32[2,3]\n"
                "  #o0 output @result = %3\n"
                "}\n");
}

TK_TEST("Graph lookups reject out-of-range and foreign IDs") {
  GraphBuilder first_builder;
  const ValueId first_value =
      require_value(first_builder.input("x", make_type({1})));
  require_output(first_builder.output("result", first_value));
  const VerifiedGraph first =
      require_graph(std::move(first_builder).finish());

  GraphBuilder second_builder;
  const ValueId second_value =
      require_value(second_builder.input("x", make_type({1})));
  require_output(second_builder.output("result", second_value));
  const VerifiedGraph second =
      require_graph(std::move(second_builder).finish());

  TK_REQUIRE(first.node(NodeId{1U}) == nullptr);
  TK_REQUIRE(first.type(second_value) == nullptr);
  TK_REQUIRE(second.type(first_value) == nullptr);
  TK_REQUIRE_EQ(require_type(first, first_value).to_string(), "f32[1]");
}

TK_TEST("Finalization requires an output and consumes on success") {
  GraphBuilder builder;
  const ValueId input = require_value(builder.input("x", make_type({2})));
  require_error(std::move(builder).finish(), ErrorCode::graph_has_no_outputs);
  require_output(builder.output("result", input));
  const auto graph = std::move(builder).finish();
  TK_REQUIRE(graph.has_value());
  require_error(builder.relu(input), ErrorCode::builder_finished);
  require_error(std::move(builder).finish(), ErrorCode::builder_finished);
}

}  // namespace
