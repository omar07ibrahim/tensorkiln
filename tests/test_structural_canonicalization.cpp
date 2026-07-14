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

#include "tensorkiln/reference.hpp"
#include "tensorkiln/structural_canonicalization.hpp"

namespace {

using tensorkiln::AddOp;
using tensorkiln::GraphBuilder;
using tensorkiln::InputBinding;
using tensorkiln::MatMulOp;
using tensorkiln::NodeProvenance;
using tensorkiln::OutputId;
using tensorkiln::ReferenceInterpreter;
using tensorkiln::ReferenceResult;
using tensorkiln::ReluOp;
using tensorkiln::Shape;
using tensorkiln::StructuralCanonicalization;
using tensorkiln::StructuralCanonicalizationResult;
using tensorkiln::StructuralCanonicalizationStats;
using tensorkiln::Tensor;
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

[[nodiscard]] TensorType scalar_type() {
  const auto type = TensorType::create(Shape::scalar());
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

[[nodiscard]] StructuralCanonicalizationResult require_canonicalized(
    tensorkiln::Result<StructuralCanonicalizationResult> result) {
  TK_REQUIRE(result.value_if() != nullptr);
  return std::move(*result.value_if());
}

[[nodiscard]] ReferenceResult require_reference(
    tensorkiln::Result<ReferenceResult> result) {
  TK_REQUIRE(result.value_if() != nullptr);
  return std::move(*result.value_if());
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

TK_TEST("Structural canonicalization merges exact Add subexpressions with stable provenance") {
  GraphBuilder builder;
  const ValueId left =
      require_value(builder.input("left", scalar_type()));
  const ValueId right =
      require_value(builder.input("right", scalar_type()));
  const ValueId first = require_value(builder.add(left, right));
  const ValueId second = require_value(builder.add(left, right));
  const ValueId sum = require_value(builder.add(first, second));
  require_output(builder.output("sum", sum));
  const VerifiedGraph source = require_graph(std::move(builder).finish());

  const StructuralCanonicalizationResult canonicalized =
      require_canonicalized(StructuralCanonicalization::run(source));
  const StructuralCanonicalizationStats expected_stats{
      5U, 4U, 1U, 1U, 0U, 0U,
  };
  TK_REQUIRE_EQ(canonicalized.stats(), expected_stats);
  TK_REQUIRE_EQ(canonicalized.graph().nodes().size(), 4U);
  TK_REQUIRE(std::holds_alternative<AddOp>(
      canonicalized.graph().nodes()[2].operation()));
  TK_REQUIRE(std::holds_alternative<AddOp>(
      canonicalized.graph().nodes()[3].operation()));
  TK_REQUIRE_EQ(canonicalized.graph().nodes()[3].inputs()[0].ordinal(), 2U);
  TK_REQUIRE_EQ(canonicalized.graph().nodes()[3].inputs()[1].ordinal(), 2U);

  const std::span<const NodeProvenance> provenance =
      canonicalized.provenance().entries();
  TK_REQUIRE_EQ(provenance.size(), 4U);
  TK_REQUIRE_EQ(provenance[2].sources().size(), 2U);
  TK_REQUIRE_EQ(provenance[2].sources()[0].node(), source.nodes()[2].id());
  TK_REQUIRE_EQ(provenance[2].sources()[1].node(), source.nodes()[3].id());
  TK_REQUIRE(canonicalized.provenance().for_source(source.nodes()[2].id()) ==
             &provenance[2]);
  TK_REQUIRE(canonicalized.provenance().for_source(source.nodes()[3].id()) ==
             &provenance[2]);

  const std::string expected_dump =
      "tensorkiln.structural_canonicalization v0 {\n"
      "  nodes {source=5, result=4, merged=1}\n"
      "  merges {common_subexpressions=1, redundant_relus=0}\n"
      "  guards {preserved_output_distinctions=0}\n"
      "  #n0 %0 <- {source #n0 %0}\n"
      "  #n1 %1 <- {source #n1 %1}\n"
      "  #n2 %2 <- {source #n2 %2, source #n3 %3}\n"
      "  #n3 %3 <- {source #n4 %4}\n"
      "}\n";
  TK_REQUIRE_EQ(canonicalized.dump(), expected_dump);
  TK_REQUIRE_EQ(canonicalized.dump(), canonicalized.dump());
}

TK_TEST("Structural canonicalization merges exact MatMul and ReLU nodes bitwise") {
  GraphBuilder builder;
  const ValueId left =
      require_value(builder.input("left", make_type({2, 2})));
  const ValueId right =
      require_value(builder.input("right", make_type({2, 2})));
  const ValueId first_product = require_value(builder.matmul(left, right));
  const ValueId second_product = require_value(builder.matmul(left, right));
  const ValueId first_relu = require_value(builder.relu(first_product));
  const ValueId second_relu = require_value(builder.relu(second_product));
  const ValueId sum = require_value(builder.add(first_relu, second_relu));
  require_output(builder.output("sum", sum));
  const VerifiedGraph source = require_graph(std::move(builder).finish());

  const StructuralCanonicalizationResult canonicalized =
      require_canonicalized(StructuralCanonicalization::run(source));
  const StructuralCanonicalizationStats expected_stats{
      7U, 5U, 2U, 2U, 0U, 0U,
  };
  TK_REQUIRE_EQ(canonicalized.stats(), expected_stats);
  TK_REQUIRE(std::holds_alternative<MatMulOp>(
      canonicalized.graph().nodes()[2].operation()));
  TK_REQUIRE(std::holds_alternative<ReluOp>(
      canonicalized.graph().nodes()[3].operation()));
  TK_REQUIRE(std::holds_alternative<AddOp>(
      canonicalized.graph().nodes()[4].operation()));
  TK_REQUIRE_EQ(canonicalized.graph().nodes()[4].inputs()[0].ordinal(), 3U);
  TK_REQUIRE_EQ(canonicalized.graph().nodes()[4].inputs()[1].ordinal(), 3U);

  const std::array<float, 4U> left_data{{
      16777216.0F, 1.0F,
      -3.0F, 0.25F,
  }};
  const std::array<float, 4U> right_data{{
      1.0F, -2.0F,
      1.0F, 8.0F,
  }};
  const std::array<InputBinding, 2U> bindings{{
      InputBinding{"left", left_data},
      InputBinding{"right", right_data},
  }};
  const ReferenceResult source_result = require_reference(
      ReferenceInterpreter::run(source, bindings));
  const ReferenceResult canonical_result = require_reference(
      ReferenceInterpreter::run(canonicalized.graph(), bindings));
  require_bits_equal(require_output_tensor(source_result, "sum").data(),
                     require_output_tensor(canonical_result, "sum").data());
}

TK_TEST("Structural canonicalization removes redundant ReLU across IEEE edge values") {
  GraphBuilder builder;
  const ValueId input =
      require_value(builder.input("x", make_type({8})));
  const ValueId first = require_value(builder.relu(input));
  const ValueId second = require_value(builder.relu(first));
  require_output(builder.output("result", second));
  const VerifiedGraph source = require_graph(std::move(builder).finish());

  const StructuralCanonicalizationResult canonicalized =
      require_canonicalized(StructuralCanonicalization::run(source));
  const StructuralCanonicalizationStats expected_stats{
      3U, 2U, 1U, 0U, 1U, 0U,
  };
  TK_REQUIRE_EQ(canonicalized.stats(), expected_stats);
  TK_REQUIRE_EQ(canonicalized.graph().outputs()[0].value().ordinal(), 1U);
  TK_REQUIRE_EQ(canonicalized.provenance().entries()[1].sources().size(), 2U);

  const std::array<std::uint32_t, 8U> input_bits{{
      UINT32_C(0xff800000), UINT32_C(0x80000001),
      UINT32_C(0x80000000), UINT32_C(0x00000000),
      UINT32_C(0x00000001), UINT32_C(0x7f800000),
      UINT32_C(0x7fc12345), UINT32_C(0x7f812345),
  }};
  std::array<float, 8U> input_data{};
  for (std::size_t index = 0U; index < input_data.size(); ++index) {
    input_data[index] = std::bit_cast<float>(input_bits[index]);
  }
  const std::array<InputBinding, 1U> bindings{{
      InputBinding{"x", input_data},
  }};
  const ReferenceResult source_result = require_reference(
      ReferenceInterpreter::run(source, bindings));
  const ReferenceResult canonical_result = require_reference(
      ReferenceInterpreter::run(canonicalized.graph(), bindings));
  require_bits_equal(require_output_tensor(source_result, "result").data(),
                     require_output_tensor(canonical_result, "result").data());
}

TK_TEST("Structural canonicalization preserves exact output alias classes") {
  GraphBuilder builder;
  const ValueId left = require_value(builder.input("left", scalar_type()));
  const ValueId right = require_value(builder.input("right", scalar_type()));
  const ValueId first = require_value(builder.add(left, right));
  const ValueId second = require_value(builder.add(left, right));
  require_output(builder.output("first_a", first));
  require_output(builder.output("first_b", first));
  require_output(builder.output("second_a", second));
  require_output(builder.output("second_b", second));
  const VerifiedGraph source = require_graph(std::move(builder).finish());

  const StructuralCanonicalizationResult canonicalized =
      require_canonicalized(StructuralCanonicalization::run(source));
  const StructuralCanonicalizationStats expected_stats{
      4U, 4U, 0U, 0U, 0U, 1U,
  };
  TK_REQUIRE_EQ(canonicalized.stats(), expected_stats);
  TK_REQUIRE_EQ(canonicalized.graph().outputs()[0].value().ordinal(), 2U);
  TK_REQUIRE_EQ(canonicalized.graph().outputs()[1].value().ordinal(), 2U);
  TK_REQUIRE_EQ(canonicalized.graph().outputs()[2].value().ordinal(), 3U);
  TK_REQUIRE_EQ(canonicalized.graph().outputs()[3].value().ordinal(), 3U);

  const std::array<float, 1U> left_data{{1.25F}};
  const std::array<float, 1U> right_data{{-0.5F}};
  const std::array<InputBinding, 2U> bindings{{
      InputBinding{"left", left_data},
      InputBinding{"right", right_data},
  }};
  const ReferenceResult source_result = require_reference(
      ReferenceInterpreter::run(source, bindings));
  const ReferenceResult canonical_result = require_reference(
      ReferenceInterpreter::run(canonicalized.graph(), bindings));
  TK_REQUIRE(source_result.output("first_a") ==
             source_result.output("first_b"));
  TK_REQUIRE(source_result.output("second_a") ==
             source_result.output("second_b"));
  TK_REQUIRE(source_result.output("first_a") !=
             source_result.output("second_a"));
  TK_REQUIRE(canonical_result.output("first_a") ==
             canonical_result.output("first_b"));
  TK_REQUIRE(canonical_result.output("second_a") ==
             canonical_result.output("second_b"));
  TK_REQUIRE(canonical_result.output("first_a") !=
             canonical_result.output("second_a"));
}

TK_TEST("Structural canonicalization protects distinct nested ReLU outputs") {
  GraphBuilder builder;
  const ValueId input = require_value(builder.input("x", scalar_type()));
  const ValueId first = require_value(builder.relu(input));
  const ValueId second = require_value(builder.relu(first));
  require_output(builder.output("first", first));
  require_output(builder.output("second", second));
  const VerifiedGraph source = require_graph(std::move(builder).finish());

  const StructuralCanonicalizationResult canonicalized =
      require_canonicalized(StructuralCanonicalization::run(source));
  const StructuralCanonicalizationStats expected_stats{
      3U, 3U, 0U, 0U, 0U, 1U,
  };
  TK_REQUIRE_EQ(canonicalized.stats(), expected_stats);
  TK_REQUIRE(canonicalized.graph().outputs()[0].value() !=
             canonicalized.graph().outputs()[1].value());

  const std::array<float, 1U> input_data{{-0.0F}};
  const std::array<InputBinding, 1U> bindings{{
      InputBinding{"x", input_data},
  }};
  const ReferenceResult canonical_result = require_reference(
      ReferenceInterpreter::run(canonicalized.graph(), bindings));
  TK_REQUIRE(canonical_result.output("first") != nullptr);
  TK_REQUIRE(canonical_result.output("second") != nullptr);
  TK_REQUIRE(canonical_result.output("first") !=
             canonical_result.output("second"));
  require_bits_equal(canonical_result.output("first")->data(),
                     canonical_result.output("second")->data());
}

TK_TEST("Structural canonicalization chooses the first output-compatible candidate") {
  GraphBuilder builder;
  const ValueId left = require_value(builder.input("left", scalar_type()));
  const ValueId right = require_value(builder.input("right", scalar_type()));
  const ValueId first = require_value(builder.add(left, right));
  const ValueId first_output = require_value(builder.add(left, right));
  const ValueId second_output = require_value(builder.add(left, right));
  const ValueId tail_input = require_value(builder.add(left, right));
  const ValueId tail = require_value(builder.relu(tail_input));
  require_output(builder.output("first", first_output));
  require_output(builder.output("second", second_output));
  require_output(builder.output("tail", tail));
  const VerifiedGraph source = require_graph(std::move(builder).finish());

  const StructuralCanonicalizationResult canonicalized =
      require_canonicalized(StructuralCanonicalization::run(source));
  const StructuralCanonicalizationStats expected_stats{
      7U, 5U, 2U, 2U, 0U, 1U,
  };
  TK_REQUIRE_EQ(canonicalized.stats(), expected_stats);
  TK_REQUIRE_EQ(canonicalized.graph().outputs()[0].value().ordinal(), 2U);
  TK_REQUIRE_EQ(canonicalized.graph().outputs()[1].value().ordinal(), 3U);
  TK_REQUIRE_EQ(canonicalized.graph().outputs()[2].value().ordinal(), 4U);
  TK_REQUIRE(first != first_output);

  const std::span<const NodeProvenance> provenance =
      canonicalized.provenance().entries();
  TK_REQUIRE_EQ(provenance[2].sources().size(), 3U);
  TK_REQUIRE_EQ(provenance[2].sources()[0].node(), source.nodes()[2].id());
  TK_REQUIRE_EQ(provenance[2].sources()[1].node(), source.nodes()[3].id());
  TK_REQUIRE_EQ(provenance[2].sources()[2].node(), source.nodes()[5].id());
  TK_REQUIRE_EQ(provenance[3].sources().size(), 1U);
  TK_REQUIRE_EQ(provenance[3].sources()[0].node(), source.nodes()[4].id());
  TK_REQUIRE_EQ(provenance[4].sources().size(), 1U);
  TK_REQUIRE_EQ(provenance[4].sources()[0].node(), source.nodes()[6].id());
}

}  // namespace
