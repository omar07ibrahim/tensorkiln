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
#include "tensorkiln/structural_canonicalization.hpp"

namespace {

using tensorkiln::AddOp;
using tensorkiln::ConstantOp;
using tensorkiln::DeadCodeElimination;
using tensorkiln::DeadCodeEliminationResult;
using tensorkiln::ElementType;
using tensorkiln::ErrorCode;
using tensorkiln::GraphBuilder;
using tensorkiln::GraphLimits;
using tensorkiln::GraphProvenance;
using tensorkiln::InputBinding;
using tensorkiln::InputOp;
using tensorkiln::MatMulOp;
using tensorkiln::NodeId;
using tensorkiln::OutputId;
using tensorkiln::ReferenceInterpreter;
using tensorkiln::ReferenceResult;
using tensorkiln::ReluOp;
using tensorkiln::Shape;
using tensorkiln::ShapeLimits;
using tensorkiln::StructuralCanonicalization;
using tensorkiln::StructuralCanonicalizationResult;
using tensorkiln::StructuralCanonicalizationStats;
using tensorkiln::Tensor;
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

[[nodiscard]] DeadCodeEliminationResult require_dce(
    tensorkiln::Result<DeadCodeEliminationResult> result) {
  TK_REQUIRE(result.value_if() != nullptr);
  return std::move(*result.value_if());
}

[[nodiscard]] GraphProvenance require_provenance(
    tensorkiln::Result<GraphProvenance> result) {
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

[[nodiscard]] VerifiedGraph build_chain(const std::string& prefix) {
  GraphBuilder builder;
  const ValueId input =
      require_value(builder.input(prefix + "_input", scalar_type()));
  const ValueId result = require_value(builder.relu(input));
  require_output(builder.output(prefix + "_output", result));
  return require_graph(std::move(builder).finish());
}

TK_TEST("Structural canonicalization preserves operand order and operation kind") {
  GraphBuilder builder;
  const ValueId left =
      require_value(builder.input("left", make_type({2, 2})));
  const ValueId right =
      require_value(builder.input("right", make_type({2, 2})));
  const ValueId add_lr = require_value(builder.add(left, right));
  const ValueId add_rl = require_value(builder.add(right, left));
  const ValueId matmul_lr = require_value(builder.matmul(left, right));
  const ValueId matmul_rl = require_value(builder.matmul(right, left));
  const ValueId add_sum = require_value(builder.add(add_lr, add_rl));
  const ValueId matmul_sum =
      require_value(builder.add(matmul_lr, matmul_rl));
  const ValueId result = require_value(builder.add(add_sum, matmul_sum));
  require_output(builder.output("result", result));
  const VerifiedGraph source = require_graph(std::move(builder).finish());

  const StructuralCanonicalizationResult canonicalized =
      require_canonicalized(StructuralCanonicalization::run(source));
  const StructuralCanonicalizationStats expected_stats{
      9U, 9U, 0U, 0U, 0U, 0U,
  };
  TK_REQUIRE_EQ(canonicalized.stats(), expected_stats);
  TK_REQUIRE_EQ(canonicalized.graph().dump(), source.dump());
  TK_REQUIRE(std::holds_alternative<AddOp>(
      canonicalized.graph().nodes()[2].operation()));
  TK_REQUIRE(std::holds_alternative<MatMulOp>(
      canonicalized.graph().nodes()[4].operation()));

  const std::array<float, 4U> left_data{{1.0F, 2.0F, 3.0F, 4.0F}};
  const std::array<float, 4U> right_data{{0.5F, -1.0F, 2.0F, 0.25F}};
  const std::array<InputBinding, 2U> bindings{{
      InputBinding{"left", left_data},
      InputBinding{"right", right_data},
  }};
  const ReferenceResult source_result = require_reference(
      ReferenceInterpreter::run(source, bindings));
  const ReferenceResult canonical_result = require_reference(
      ReferenceInterpreter::run(canonicalized.graph(), bindings));
  require_bits_equal(require_output_tensor(source_result, "result").data(),
                     require_output_tensor(canonical_result, "result").data());
}

TK_TEST("Structural canonicalization keeps Inputs and raw Constants distinct") {
  GraphBuilder builder;
  const TensorType type = make_type({5});
  const ValueId left = require_value(builder.input("left", type));
  const ValueId right = require_value(builder.input("right", type));
  const std::array<std::uint32_t, 5U> payload_bits{{
      UINT32_C(0x00000000), UINT32_C(0x80000000),
      UINT32_C(0x00000001), UINT32_C(0x7fc12345),
      UINT32_C(0x7f812345),
  }};
  std::array<float, 5U> payload{};
  for (std::size_t index = 0U; index < payload.size(); ++index) {
    payload[index] = std::bit_cast<float>(payload_bits[index]);
  }
  const ValueId first_constant =
      require_value(builder.constant("first_constant", type, payload));
  const ValueId second_constant =
      require_value(builder.constant("second_constant", type, payload));
  const ValueId first_sum =
      require_value(builder.add(left, first_constant));
  const ValueId second_sum =
      require_value(builder.add(right, second_constant));
  const ValueId result = require_value(builder.add(first_sum, second_sum));
  require_output(builder.output("result", result));
  const VerifiedGraph source = require_graph(std::move(builder).finish());

  const StructuralCanonicalizationResult canonicalized =
      require_canonicalized(StructuralCanonicalization::run(source));
  const StructuralCanonicalizationStats expected_stats{
      7U, 7U, 0U, 0U, 0U, 0U,
  };
  TK_REQUIRE_EQ(canonicalized.stats(), expected_stats);
  TK_REQUIRE(std::holds_alternative<InputOp>(
      canonicalized.graph().nodes()[0].operation()));
  TK_REQUIRE(std::holds_alternative<InputOp>(
      canonicalized.graph().nodes()[1].operation()));
  TK_REQUIRE_EQ(std::get<InputOp>(canonicalized.graph().nodes()[0].operation())
                    .name,
                "left");
  TK_REQUIRE_EQ(std::get<InputOp>(canonicalized.graph().nodes()[1].operation())
                    .name,
                "right");

  for (std::size_t node_index = 2U; node_index < 4U; ++node_index) {
    const ConstantOp& source_constant =
        std::get<ConstantOp>(source.nodes()[node_index].operation());
    const ConstantOp& result_constant = std::get<ConstantOp>(
        canonicalized.graph().nodes()[node_index].operation());
    TK_REQUIRE_EQ(result_constant.name, source_constant.name);
    TK_REQUIRE_EQ(result_constant.fingerprint, source_constant.fingerprint);
    require_bits_equal(result_constant.data, source_constant.data);
  }
  TK_REQUIRE(canonicalized.graph().nodes()[2].output() !=
             canonicalized.graph().nodes()[3].output());
}

TK_TEST("Structural canonicalization never removes unique dead work") {
  GraphBuilder builder;
  const TensorType type = make_type({2, 2});
  const ValueId left = require_value(builder.input("left", type));
  const ValueId right = require_value(builder.input("right", type));
  const std::array<float, 4U> dead_data{{1.0F, 2.0F, 3.0F, 4.0F}};
  const ValueId dead_constant =
      require_value(builder.constant("dead_constant", type, dead_data));
  const ValueId dead_relu = require_value(builder.relu(left));
  const ValueId dead_matmul = require_value(builder.matmul(left, right));
  const ValueId unique_dead_add = require_value(builder.add(left, left));
  const ValueId first_dead_add = require_value(builder.add(left, right));
  const ValueId second_dead_add = require_value(builder.add(left, right));
  require_output(builder.output("right", right));
  const VerifiedGraph source = require_graph(std::move(builder).finish());

  const StructuralCanonicalizationResult canonicalized =
      require_canonicalized(StructuralCanonicalization::run(source));
  const StructuralCanonicalizationStats expected_stats{
      8U, 7U, 1U, 1U, 0U, 0U,
  };
  TK_REQUIRE_EQ(canonicalized.stats(), expected_stats);
  TK_REQUIRE(std::holds_alternative<ConstantOp>(
      canonicalized.graph().nodes()[2].operation()));
  TK_REQUIRE(std::holds_alternative<ReluOp>(
      canonicalized.graph().nodes()[3].operation()));
  TK_REQUIRE(std::holds_alternative<MatMulOp>(
      canonicalized.graph().nodes()[4].operation()));
  TK_REQUIRE(std::holds_alternative<AddOp>(
      canonicalized.graph().nodes()[5].operation()));
  TK_REQUIRE(std::holds_alternative<AddOp>(
      canonicalized.graph().nodes()[6].operation()));

  const auto* dead_constant_provenance =
      canonicalized.provenance().for_source(dead_constant);
  const auto* dead_relu_provenance =
      canonicalized.provenance().for_source(dead_relu);
  const auto* dead_matmul_provenance =
      canonicalized.provenance().for_source(dead_matmul);
  const auto* unique_add_provenance =
      canonicalized.provenance().for_source(unique_dead_add);
  const auto* first_add_provenance =
      canonicalized.provenance().for_source(first_dead_add);
  const auto* second_add_provenance =
      canonicalized.provenance().for_source(second_dead_add);
  TK_REQUIRE(dead_constant_provenance != nullptr);
  TK_REQUIRE(dead_relu_provenance != nullptr);
  TK_REQUIRE(dead_matmul_provenance != nullptr);
  TK_REQUIRE(unique_add_provenance != nullptr);
  TK_REQUIRE(first_add_provenance != nullptr);
  TK_REQUIRE(second_add_provenance != nullptr);
  TK_REQUIRE_EQ(dead_constant_provenance->result_node(),
                canonicalized.graph().nodes()[2].id());
  TK_REQUIRE_EQ(dead_relu_provenance->result_node(),
                canonicalized.graph().nodes()[3].id());
  TK_REQUIRE_EQ(dead_matmul_provenance->result_node(),
                canonicalized.graph().nodes()[4].id());
  TK_REQUIRE_EQ(unique_add_provenance->result_node(),
                canonicalized.graph().nodes()[5].id());
  TK_REQUIRE(first_add_provenance == second_add_provenance);
  TK_REQUIRE(first_add_provenance != unique_add_provenance);
}

TK_TEST("Structural canonicalization refuses algebraic and floating rewrites") {
  GraphBuilder builder;
  const TensorType type = make_type({2, 2});
  const ValueId input = require_value(builder.input("input", type));
  const std::array<float, 4U> zero_data{{0.0F, -0.0F, 0.0F, -0.0F}};
  const ValueId zero =
      require_value(builder.constant("signed_zero", type, zero_data));
  const std::array<float, 4U> identity_data{{
      1.0F, 0.0F,
      0.0F, 1.0F,
  }};
  const ValueId identity =
      require_value(builder.constant("identity", type, identity_data));

  const ValueId add_zero = require_value(builder.add(input, zero));
  const ValueId left_associated = require_value(builder.add(add_zero, zero));
  const ValueId zero_sum = require_value(builder.add(zero, zero));
  const ValueId right_associated = require_value(builder.add(input, zero_sum));
  const ValueId matmul_identity =
      require_value(builder.matmul(input, identity));
  const ValueId matmul_zero = require_value(builder.matmul(input, zero));
  const ValueId unfused =
      require_value(builder.add(matmul_identity, zero));
  const ValueId zero_product =
      require_value(builder.matmul(zero, identity));
  const ValueId distribution_source =
      require_value(builder.matmul(add_zero, identity));
  const ValueId distribution_expansion =
      require_value(builder.add(matmul_identity, zero_product));
  const ValueId first_half =
      require_value(builder.add(left_associated, right_associated));
  const ValueId second_half = require_value(builder.add(matmul_zero, unfused));
  const ValueId algebraic_half =
      require_value(builder.add(first_half, second_half));
  const ValueId distribution_half = require_value(
      builder.add(distribution_source, distribution_expansion));
  const ValueId result =
      require_value(builder.add(algebraic_half, distribution_half));
  require_output(builder.output("result", result));
  const VerifiedGraph source = require_graph(std::move(builder).finish());

  const StructuralCanonicalizationResult canonicalized =
      require_canonicalized(StructuralCanonicalization::run(source));
  const StructuralCanonicalizationStats expected_stats{
      18U, 18U, 0U, 0U, 0U, 0U,
  };
  TK_REQUIRE_EQ(canonicalized.stats(), expected_stats);
  TK_REQUIRE_EQ(canonicalized.graph().dump(), source.dump());
}

TK_TEST("Structural canonicalization preserves exact custom graph limits") {
  GraphLimits limits;
  limits.max_nodes = 4U;
  limits.max_outputs = 1U;
  limits.max_name_bytes = 24U;
  limits.max_constant_elements = 0U;
  limits.shape_limits = ShapeLimits{8U};
  limits.tensor_limits = TensorLimits{32U};
  GraphBuilder builder(limits);
  const TensorType type =
      make_type({2}, limits.shape_limits, limits.tensor_limits);
  const ValueId input = require_value(builder.input("input", type));
  const ValueId first = require_value(builder.relu(input));
  const ValueId second = require_value(builder.relu(input));
  const ValueId result = require_value(builder.add(first, second));
  require_output(builder.output("result", result));
  const VerifiedGraph source = require_graph(std::move(builder).finish());

  const StructuralCanonicalizationResult canonicalized =
      require_canonicalized(StructuralCanonicalization::run(source));
  const StructuralCanonicalizationStats expected_stats{
      4U, 3U, 1U, 1U, 0U, 0U,
  };
  TK_REQUIRE_EQ(canonicalized.stats(), expected_stats);
  TK_REQUIRE_EQ(canonicalized.graph().limits(), limits);
}

TK_TEST("DCE and structural canonicalization compose lineage to root definitions") {
  GraphBuilder builder;
  const ValueId left = require_value(builder.input("left", scalar_type()));
  const ValueId right = require_value(builder.input("right", scalar_type()));
  const std::array<float, 1U> dead_data{{-1.0F}};
  const ValueId dead_constant = require_value(
      builder.constant("dead_constant", scalar_type(), dead_data));
  const ValueId first_add = require_value(builder.add(left, right));
  const ValueId second_add = require_value(builder.add(left, right));
  const ValueId dead_relu = require_value(builder.relu(dead_constant));
  const ValueId first_relu = require_value(builder.relu(first_add));
  const ValueId second_relu = require_value(builder.relu(second_add));
  const ValueId result = require_value(builder.add(first_relu, second_relu));
  require_output(builder.output("result", result));
  const VerifiedGraph root = require_graph(std::move(builder).finish());

  const DeadCodeEliminationResult eliminated =
      require_dce(DeadCodeElimination::run(root));
  TK_REQUIRE_EQ(eliminated.graph().nodes().size(), 7U);
  const StructuralCanonicalizationResult canonicalized =
      require_canonicalized(StructuralCanonicalization::run(
          eliminated.graph(), eliminated.provenance()));
  const StructuralCanonicalizationStats expected_stats{
      7U, 5U, 2U, 2U, 0U, 0U,
  };
  TK_REQUIRE_EQ(canonicalized.stats(), expected_stats);

  const std::string expected_provenance =
      "tensorkiln.provenance v0 {\n"
      "  #n0 %0 <- {source #n0 %0}\n"
      "  #n1 %1 <- {source #n1 %1}\n"
      "  #n2 %2 <- {source #n3 %3, source #n4 %4}\n"
      "  #n3 %3 <- {source #n6 %6, source #n7 %7}\n"
      "  #n4 %4 <- {source #n8 %8}\n"
      "}\n";
  TK_REQUIRE_EQ(canonicalized.provenance().dump(), expected_provenance);
  TK_REQUIRE(canonicalized.provenance().for_source(dead_constant) == nullptr);
  TK_REQUIRE(canonicalized.provenance().for_source(dead_relu) == nullptr);
  const auto* first_add_provenance =
      canonicalized.provenance().for_source(root.nodes()[3].id());
  const auto* second_add_provenance =
      canonicalized.provenance().for_source(root.nodes()[4].id());
  const auto* first_relu_provenance =
      canonicalized.provenance().for_source(root.nodes()[6].id());
  const auto* second_relu_provenance =
      canonicalized.provenance().for_source(root.nodes()[7].id());
  const auto* result_provenance =
      canonicalized.provenance().for_source(root.nodes()[8].id());
  TK_REQUIRE(first_add_provenance != nullptr);
  TK_REQUIRE(second_add_provenance != nullptr);
  TK_REQUIRE(first_relu_provenance != nullptr);
  TK_REQUIRE(second_relu_provenance != nullptr);
  TK_REQUIRE(result_provenance != nullptr);
  TK_REQUIRE(first_add_provenance == second_add_provenance);
  TK_REQUIRE(first_relu_provenance == second_relu_provenance);
  TK_REQUIRE_EQ(first_add_provenance->result_value(),
                canonicalized.graph().nodes()[2].output());
  TK_REQUIRE_EQ(first_relu_provenance->result_value(),
                canonicalized.graph().nodes()[3].output());
  TK_REQUIRE_EQ(result_provenance->result_value(),
                canonicalized.graph().nodes()[4].output());

  const std::array<float, 1U> left_data{{-2.0F}};
  const std::array<float, 1U> right_data{{3.5F}};
  const std::array<InputBinding, 2U> bindings{{
      InputBinding{"left", left_data},
      InputBinding{"right", right_data},
  }};
  const ReferenceResult root_result = require_reference(
      ReferenceInterpreter::run(root, bindings));
  const ReferenceResult canonical_result = require_reference(
      ReferenceInterpreter::run(canonicalized.graph(), bindings));
  require_bits_equal(require_output_tensor(root_result, "result").data(),
                     require_output_tensor(canonical_result, "result").data());
}

TK_TEST("Structural canonicalization is deterministic and graph-idempotent") {
  GraphBuilder builder;
  const ValueId left = require_value(builder.input("left", scalar_type()));
  const ValueId right = require_value(builder.input("right", scalar_type()));
  const ValueId add0 = require_value(builder.add(left, right));
  const ValueId add1 = require_value(builder.add(left, right));
  const ValueId add2 = require_value(builder.add(left, right));
  const ValueId relu0 = require_value(builder.relu(add0));
  const ValueId relu1 = require_value(builder.relu(add1));
  const ValueId relu2 = require_value(builder.relu(relu1));
  const ValueId result = require_value(builder.add(relu0, relu2));
  require_output(builder.output("result", result));
  const VerifiedGraph source = require_graph(std::move(builder).finish());
  TK_REQUIRE(add1 != add2);

  const StructuralCanonicalizationResult first =
      require_canonicalized(StructuralCanonicalization::run(source));
  const StructuralCanonicalizationResult repeated =
      require_canonicalized(StructuralCanonicalization::run(source));
  const StructuralCanonicalizationStats expected_first_stats{
      9U, 5U, 4U, 3U, 1U, 0U,
  };
  TK_REQUIRE_EQ(first.stats(), expected_first_stats);
  TK_REQUIRE_EQ(repeated.stats(), first.stats());
  TK_REQUIRE_EQ(repeated.graph().dump(), first.graph().dump());
  TK_REQUIRE_EQ(repeated.provenance().dump(), first.provenance().dump());
  TK_REQUIRE_EQ(repeated.dump(), first.dump());

  const StructuralCanonicalizationResult idempotent =
      require_canonicalized(StructuralCanonicalization::run(
          first.graph(), first.provenance()));
  const StructuralCanonicalizationStats expected_second_stats{
      5U, 5U, 0U, 0U, 0U, 0U,
  };
  TK_REQUIRE_EQ(idempotent.stats(), expected_second_stats);
  TK_REQUIRE_EQ(idempotent.graph().dump(), first.graph().dump());
  TK_REQUIRE_EQ(idempotent.provenance().dump(), first.provenance().dump());
}

TK_TEST("Output-protected canonical graphs remain stable on a second pass") {
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

  const StructuralCanonicalizationResult first_pass =
      require_canonicalized(StructuralCanonicalization::run(source));
  const StructuralCanonicalizationStats expected_stats{
      4U, 4U, 0U, 0U, 0U, 1U,
  };
  TK_REQUIRE_EQ(first_pass.stats(), expected_stats);
  const StructuralCanonicalizationResult second_pass =
      require_canonicalized(StructuralCanonicalization::run(
          first_pass.graph(), first_pass.provenance()));
  TK_REQUIRE_EQ(second_pass.stats(), expected_stats);
  TK_REQUIRE_EQ(second_pass.graph().dump(), first_pass.graph().dump());
  TK_REQUIRE_EQ(second_pass.provenance().dump(),
                first_pass.provenance().dump());
}

TK_TEST("Structural canonicalization rejects incomplete and foreign provenance") {
  const VerifiedGraph source = build_chain("source");

  GraphBuilder one_builder;
  const ValueId one_input =
      require_value(one_builder.input("one_input", scalar_type()));
  require_output(one_builder.output("one_output", one_input));
  const VerifiedGraph one =
      require_graph(std::move(one_builder).finish());
  std::vector<std::vector<NodeId>> incomplete_sources{
      {source.nodes()[0].id()},
  };
  const GraphProvenance incomplete = require_provenance(
      GraphProvenance::create(one, source, std::move(incomplete_sources)));
  const auto incomplete_result =
      StructuralCanonicalization::run(source, incomplete);
  const tensorkiln::Diagnostic& incomplete_error = require_error(
      incomplete_result, ErrorCode::provenance_domain_mismatch);
  TK_REQUIRE_EQ(incomplete_error.message,
                "source provenance has 1 entries for a graph with 2 nodes");

  const VerifiedGraph foreign = build_chain("foreign");
  const VerifiedGraph roots = build_chain("roots");
  std::vector<std::vector<NodeId>> foreign_sources{
      {roots.nodes()[0].id()},
      {roots.nodes()[1].id()},
  };
  const GraphProvenance foreign_provenance = require_provenance(
      GraphProvenance::create(
          foreign, roots, std::move(foreign_sources)));
  const auto foreign_result =
      StructuralCanonicalization::run(source, foreign_provenance);
  const tensorkiln::Diagnostic& foreign_error = require_error(
      foreign_result, ErrorCode::provenance_domain_mismatch);
  TK_REQUIRE_EQ(foreign_error.message,
                "source provenance does not describe #n0 %0");
}

}  // namespace
