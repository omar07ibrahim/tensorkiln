#include "test.hpp"

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "tensorkiln/reference.hpp"
#include "tensorkiln/structural_canonicalization.hpp"

namespace {

using tensorkiln::AddOp;
using tensorkiln::GraphBuilder;
using tensorkiln::GraphLimits;
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

  [[nodiscard]] std::size_t index(const std::size_t bound) noexcept {
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
  std::vector<std::array<ValueId, 2U>> add_duplicates;
  std::vector<std::array<ValueId, 2U>> relu_duplicates;
  std::vector<std::array<ValueId, 2U>> redundant_relus;
  std::vector<std::array<ValueId, 2U>> matmul_duplicates;
};

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

void require_bits_equal(const std::span<const float> left,
                        const std::span<const float> right) {
  TK_REQUIRE_EQ(left.size(), right.size());
  for (std::size_t index = 0U; index < left.size(); ++index) {
    TK_REQUIRE_EQ(std::bit_cast<std::uint32_t>(left[index]),
                  std::bit_cast<std::uint32_t>(right[index]));
  }
}

[[nodiscard]] SeededFixture build_seeded_graph(const std::uint32_t seed,
                                               const bool matrix) {
  DeterministicGenerator generator(seed);
  GraphBuilder builder;
  const TensorType type = matrix ? make_type({2, 2}) : scalar_type();
  std::vector<ValueId> values;
  values.reserve(matrix ? 40U : 36U);
  values.push_back(require_value(builder.input("left", type)));
  values.push_back(require_value(builder.input("right", type)));

  if (matrix) {
    const std::array<float, 4U> first_data{{
        1.0F, -2.0F,
        0.5F, -0.0F,
    }};
    const std::array<float, 4U> second_data{{
        -0.25F, 3.0F,
        2.0F, 1.0F,
    }};
    values.push_back(require_value(
        builder.constant("first_constant", type, first_data)));
    values.push_back(require_value(
        builder.constant("second_constant", type, second_data)));
  } else {
    const std::array<float, 1U> first_data{{-0.0F}};
    const std::array<float, 1U> second_data{{0.375F}};
    values.push_back(require_value(
        builder.constant("first_constant", type, first_data)));
    values.push_back(require_value(
        builder.constant("second_constant", type, second_data)));
  }

  std::optional<ValueId> protected_add_first;
  std::optional<ValueId> protected_add_second;
  std::optional<ValueId> protected_matmul_first;
  std::optional<ValueId> protected_matmul_second;
  std::vector<std::array<ValueId, 2U>> add_duplicates;
  std::vector<std::array<ValueId, 2U>> relu_duplicates;
  std::vector<std::array<ValueId, 2U>> redundant_relus;
  std::vector<std::array<ValueId, 2U>> matmul_duplicates;
  const std::size_t group_count = matrix ? 4U : 6U;

  for (std::size_t group = 0U; group < group_count; ++group) {
    const ValueId add_left = values[generator.index(values.size())];
    const ValueId add_right = values[generator.index(values.size())];
    const ValueId first_add =
        require_value(builder.add(add_left, add_right));
    const ValueId second_add =
        require_value(builder.add(add_left, add_right));
    add_duplicates.push_back(
        std::array<ValueId, 2U>{first_add, second_add});
    values.push_back(first_add);
    values.push_back(second_add);
    if (group == 0U) {
      protected_add_first = first_add;
      protected_add_second = second_add;
    }

    const ValueId first_relu = require_value(builder.relu(first_add));
    const ValueId second_relu = require_value(builder.relu(first_add));
    const ValueId nested_relu = require_value(builder.relu(first_relu));
    relu_duplicates.push_back(
        std::array<ValueId, 2U>{first_relu, second_relu});
    redundant_relus.push_back(
        std::array<ValueId, 2U>{first_relu, nested_relu});
    values.push_back(first_relu);
    values.push_back(second_relu);
    values.push_back(nested_relu);

    if (matrix) {
      const ValueId matmul_left = values[generator.index(values.size())];
      const ValueId matmul_right = values[generator.index(values.size())];
      const ValueId first_matmul =
          require_value(builder.matmul(matmul_left, matmul_right));
      const ValueId second_matmul =
          require_value(builder.matmul(matmul_left, matmul_right));
      matmul_duplicates.push_back(
          std::array<ValueId, 2U>{first_matmul, second_matmul});
      values.push_back(first_matmul);
      values.push_back(second_matmul);
      if (group == 0U) {
        protected_matmul_first = first_matmul;
        protected_matmul_second = second_matmul;
      }
    }
  }

  TK_REQUIRE(protected_add_first.has_value());
  TK_REQUIRE(protected_add_second.has_value());
  require_output(builder.output("add_first_a", *protected_add_first));
  require_output(builder.output("add_first_b", *protected_add_first));
  require_output(builder.output("add_second_a", *protected_add_second));
  require_output(builder.output("add_second_b", *protected_add_second));
  if (matrix) {
    TK_REQUIRE(protected_matmul_first.has_value());
    TK_REQUIRE(protected_matmul_second.has_value());
    require_output(builder.output("matmul_first", *protected_matmul_first));
    require_output(builder.output("matmul_second", *protected_matmul_second));
  }
  require_output(builder.output("final_a", values.back()));
  require_output(builder.output("final_b", values.back()));
  return SeededFixture{
      require_graph(std::move(builder).finish()),
      std::move(add_duplicates),
      std::move(relu_duplicates),
      std::move(redundant_relus),
      std::move(matmul_duplicates),
  };
}

template <typename Operation>
void require_merged_pairs(
    const StructuralCanonicalizationResult& canonicalized,
    const std::span<const std::array<ValueId, 2U>> pairs,
    const std::size_t begin) {
  for (std::size_t index = begin; index < pairs.size(); ++index) {
    const NodeProvenance* first =
        canonicalized.provenance().for_source(pairs[index][0]);
    const NodeProvenance* second =
        canonicalized.provenance().for_source(pairs[index][1]);
    TK_REQUIRE(first != nullptr);
    TK_REQUIRE(second != nullptr);
    TK_REQUIRE(first == second);
    const auto* result_node =
        canonicalized.graph().node(first->result_node());
    TK_REQUIRE(result_node != nullptr);
    TK_REQUIRE(std::holds_alternative<Operation>(result_node->operation()));
  }
}

template <typename Operation>
void require_protected_pair(
    const StructuralCanonicalizationResult& canonicalized,
    const std::array<ValueId, 2U>& pair) {
  const NodeProvenance* first =
      canonicalized.provenance().for_source(pair[0]);
  const NodeProvenance* second =
      canonicalized.provenance().for_source(pair[1]);
  TK_REQUIRE(first != nullptr);
  TK_REQUIRE(second != nullptr);
  TK_REQUIRE(first != second);
  const auto* first_node = canonicalized.graph().node(first->result_node());
  const auto* second_node = canonicalized.graph().node(second->result_node());
  TK_REQUIRE(first_node != nullptr);
  TK_REQUIRE(second_node != nullptr);
  TK_REQUIRE(std::holds_alternative<Operation>(first_node->operation()));
  TK_REQUIRE(std::holds_alternative<Operation>(second_node->operation()));
}

void require_seeded_properties(
    const SeededFixture& fixture,
    const std::span<const InputBinding> bindings) {
  const VerifiedGraph& source = fixture.graph;
  const StructuralCanonicalizationResult first =
      require_canonicalized(StructuralCanonicalization::run(source));
  const StructuralCanonicalizationResult repeated =
      require_canonicalized(StructuralCanonicalization::run(source));
  const StructuralCanonicalizationStats& stats = first.stats();

  TK_REQUIRE_EQ(static_cast<std::size_t>(stats.source_nodes),
                source.nodes().size());
  TK_REQUIRE_EQ(static_cast<std::size_t>(stats.result_nodes),
                first.graph().nodes().size());
  TK_REQUIRE_EQ(stats.result_nodes + stats.merged_nodes,
                stats.source_nodes);
  TK_REQUIRE_EQ(stats.common_subexpressions + stats.redundant_relus,
                stats.merged_nodes);
  TK_REQUIRE(stats.common_subexpressions > 0U);
  TK_REQUIRE(stats.redundant_relus > 0U);
  TK_REQUIRE(stats.preserved_output_distinctions > 0U);
  TK_REQUIRE_EQ(first.graph().limits(), source.limits());
  TK_REQUIRE_EQ(first.graph().outputs().size(), source.outputs().size());
  TK_REQUIRE_EQ(repeated.stats(), first.stats());
  TK_REQUIRE_EQ(repeated.graph().dump(), first.graph().dump());
  TK_REQUIRE_EQ(repeated.provenance().dump(), first.provenance().dump());
  TK_REQUIRE_EQ(repeated.dump(), first.dump());

  for (std::size_t index = 0U; index < source.outputs().size(); ++index) {
    const auto& source_output = source.outputs()[index];
    const auto& result_output = first.graph().outputs()[index];
    TK_REQUIRE_EQ(source_output.id(), result_output.id());
    TK_REQUIRE_EQ(source_output.name(), result_output.name());
    const TensorType* source_type = source.type(source_output.value());
    const TensorType* result_type = first.graph().type(result_output.value());
    TK_REQUIRE(source_type != nullptr);
    TK_REQUIRE(result_type != nullptr);
    TK_REQUIRE_EQ(*source_type, *result_type);
  }
  for (std::size_t left = 0U; left < source.outputs().size(); ++left) {
    for (std::size_t right = left + 1U; right < source.outputs().size();
         ++right) {
      TK_REQUIRE_EQ(source.outputs()[left].value() ==
                        source.outputs()[right].value(),
                    first.graph().outputs()[left].value() ==
                        first.graph().outputs()[right].value());
    }
  }

  std::size_t source_count = 0U;
  for (const NodeProvenance& entry : first.provenance().entries()) {
    TK_REQUIRE(!entry.sources().empty());
    TK_REQUIRE(first.provenance().for_result(entry.result_node()) == &entry);
    TK_REQUIRE(first.provenance().for_result(entry.result_value()) == &entry);
    source_count += entry.sources().size();
  }
  TK_REQUIRE_EQ(source_count, source.nodes().size());
  for (const auto& source_node : source.nodes()) {
    const NodeProvenance* by_node =
        first.provenance().for_source(source_node.id());
    const NodeProvenance* by_value =
        first.provenance().for_source(source_node.output());
    TK_REQUIRE(by_node != nullptr);
    TK_REQUIRE(by_node == by_value);
  }

  TK_REQUIRE(!fixture.add_duplicates.empty());
  require_protected_pair<AddOp>(first, fixture.add_duplicates[0]);
  require_merged_pairs<AddOp>(first, fixture.add_duplicates, 1U);
  require_merged_pairs<ReluOp>(first, fixture.relu_duplicates, 0U);
  require_merged_pairs<ReluOp>(first, fixture.redundant_relus, 0U);
  if (!fixture.matmul_duplicates.empty()) {
    require_protected_pair<MatMulOp>(first, fixture.matmul_duplicates[0]);
    require_merged_pairs<MatMulOp>(
        first, fixture.matmul_duplicates, 1U);
  }

  const ReferenceResult source_result = require_reference(
      ReferenceInterpreter::run(source, bindings));
  const ReferenceResult canonical_result = require_reference(
      ReferenceInterpreter::run(first.graph(), bindings));
  for (const auto& output : source.outputs()) {
    const Tensor* source_tensor = source_result.output(output.name());
    const Tensor* result_tensor = canonical_result.output(output.name());
    TK_REQUIRE(source_tensor != nullptr);
    TK_REQUIRE(result_tensor != nullptr);
    require_bits_equal(source_tensor->data(), result_tensor->data());
  }
  for (const auto& source_node : source.nodes()) {
    const NodeProvenance* provenance =
        first.provenance().for_source(source_node.output());
    TK_REQUIRE(provenance != nullptr);
    const Tensor* source_tensor = source_result.value(source_node.output());
    const Tensor* result_tensor =
        canonical_result.value(provenance->result_value());
    TK_REQUIRE(source_tensor != nullptr);
    TK_REQUIRE(result_tensor != nullptr);
    require_bits_equal(source_tensor->data(), result_tensor->data());
  }
  for (std::size_t left = 0U; left < source.outputs().size(); ++left) {
    for (std::size_t right = left + 1U; right < source.outputs().size();
         ++right) {
      const std::string_view left_name = source.outputs()[left].name();
      const std::string_view right_name = source.outputs()[right].name();
      TK_REQUIRE_EQ(source_result.output(left_name) ==
                        source_result.output(right_name),
                    canonical_result.output(left_name) ==
                        canonical_result.output(right_name));
    }
  }
}

void require_owner_independence(const SeededFixture& first,
                                const SeededFixture& rebuilt) {
  TK_REQUIRE_EQ(rebuilt.graph.dump(), first.graph.dump());
  const StructuralCanonicalizationResult first_result =
      require_canonicalized(StructuralCanonicalization::run(first.graph));
  const StructuralCanonicalizationResult rebuilt_result =
      require_canonicalized(StructuralCanonicalization::run(rebuilt.graph));
  TK_REQUIRE_EQ(rebuilt_result.stats(), first_result.stats());
  TK_REQUIRE_EQ(rebuilt_result.graph().dump(), first_result.graph().dump());
  TK_REQUIRE_EQ(rebuilt_result.provenance().dump(),
                first_result.provenance().dump());
  TK_REQUIRE_EQ(rebuilt_result.dump(), first_result.dump());
}

TK_TEST("Structural canonicalization is differential across seeded scalar DAGs") {
  std::set<std::string> topologies;
  for (std::uint32_t seed = 1U; seed <= 48U; ++seed) {
    const SeededFixture fixture = build_seeded_graph(seed, false);
    const SeededFixture rebuilt = build_seeded_graph(seed, false);
    topologies.insert(fixture.graph.dump());
    require_owner_independence(fixture, rebuilt);
    const std::array<float, 1U> left_data{{
        static_cast<float>(seed % 17U) * 0.125F - 1.0F,
    }};
    const std::array<float, 1U> right_data{{
        static_cast<float>(seed % 11U) * -0.25F + 0.75F,
    }};
    const std::array<InputBinding, 2U> bindings{{
        InputBinding{"left", left_data},
        InputBinding{"right", right_data},
    }};
    require_seeded_properties(fixture, bindings);
  }
  TK_REQUIRE(topologies.size() >= 40U);
}

TK_TEST("Structural canonicalization is differential across seeded matrix DAGs") {
  std::set<std::string> topologies;
  for (std::uint32_t seed = 101U; seed <= 116U; ++seed) {
    const SeededFixture fixture = build_seeded_graph(seed, true);
    const SeededFixture rebuilt = build_seeded_graph(seed, true);
    topologies.insert(fixture.graph.dump());
    require_owner_independence(fixture, rebuilt);
    const float offset = static_cast<float>(seed % 7U) * 0.03125F;
    const std::array<float, 4U> left_data{{
        16777216.0F, 1.0F + offset,
        -3.0F, 0.25F,
    }};
    const std::array<float, 4U> right_data{{
        1.0F, -2.0F,
        1.0F - offset, 8.0F,
    }};
    const std::array<InputBinding, 2U> bindings{{
        InputBinding{"left", left_data},
        InputBinding{"right", right_data},
    }};
    require_seeded_properties(fixture, bindings);
  }
  TK_REQUIRE(topologies.size() >= 14U);
}

TK_TEST("Structural canonicalization preserves rank-four broadcasted matrix DAGs") {
  GraphBuilder builder;
  const TensorType left_type = make_type({2, 1, 2, 3});
  const TensorType right_type = make_type({1, 3, 3, 2});
  const TensorType bias_type = make_type({1, 3, 1, 2});
  const ValueId left = require_value(builder.input("left", left_type));
  const ValueId right = require_value(builder.input("right", right_type));
  const std::array<float, 6U> bias_data{{
      0.25F, -0.5F,
      1.0F, -1.5F,
      2.0F, -2.5F,
  }};
  const ValueId bias =
      require_value(builder.constant("bias", bias_type, bias_data));
  const ValueId first_matmul = require_value(builder.matmul(left, right));
  const ValueId second_matmul = require_value(builder.matmul(left, right));
  const ValueId first_add = require_value(builder.add(first_matmul, bias));
  const ValueId second_add = require_value(builder.add(second_matmul, bias));
  const ValueId first_relu = require_value(builder.relu(first_add));
  const ValueId second_relu = require_value(builder.relu(second_add));
  const ValueId nested_relu = require_value(builder.relu(second_relu));
  const ValueId result = require_value(builder.add(first_relu, nested_relu));
  require_output(builder.output("result", result));
  const VerifiedGraph source = require_graph(std::move(builder).finish());

  const StructuralCanonicalizationResult canonicalized =
      require_canonicalized(StructuralCanonicalization::run(source));
  const StructuralCanonicalizationStats expected_stats{
      11U, 7U, 4U, 3U, 1U, 0U,
  };
  TK_REQUIRE_EQ(canonicalized.stats(), expected_stats);
  const TensorType expected_type = make_type({2, 3, 2, 2});
  const TensorType* source_type = source.type(result);
  const TensorType* result_type =
      canonicalized.graph().type(canonicalized.graph().outputs()[0].value());
  TK_REQUIRE(source_type != nullptr);
  TK_REQUIRE(result_type != nullptr);
  TK_REQUIRE_EQ(*source_type, expected_type);
  TK_REQUIRE_EQ(*result_type, expected_type);

  const NodeProvenance* matmul_provenance =
      canonicalized.provenance().for_source(first_matmul);
  const NodeProvenance* second_matmul_provenance =
      canonicalized.provenance().for_source(second_matmul);
  const NodeProvenance* add_provenance =
      canonicalized.provenance().for_source(first_add);
  const NodeProvenance* second_add_provenance =
      canonicalized.provenance().for_source(second_add);
  const NodeProvenance* relu_provenance =
      canonicalized.provenance().for_source(first_relu);
  const NodeProvenance* second_relu_provenance =
      canonicalized.provenance().for_source(second_relu);
  const NodeProvenance* nested_relu_provenance =
      canonicalized.provenance().for_source(nested_relu);
  TK_REQUIRE(matmul_provenance != nullptr);
  TK_REQUIRE(add_provenance != nullptr);
  TK_REQUIRE(relu_provenance != nullptr);
  TK_REQUIRE(matmul_provenance == second_matmul_provenance);
  TK_REQUIRE(add_provenance == second_add_provenance);
  TK_REQUIRE(relu_provenance == second_relu_provenance);
  TK_REQUIRE(relu_provenance == nested_relu_provenance);

  std::array<float, 12U> left_data{};
  for (std::size_t index = 0U; index < left_data.size(); ++index) {
    left_data[index] = static_cast<float>((index % 7U) + 1U) * 0.25F;
  }
  std::array<float, 18U> right_data{};
  for (std::size_t index = 0U; index < right_data.size(); ++index) {
    const float magnitude =
        static_cast<float>((index % 5U) + 1U) * 0.125F;
    right_data[index] = index % 2U == 0U ? magnitude : -magnitude;
  }
  const std::array<InputBinding, 2U> bindings{{
      InputBinding{"left", left_data},
      InputBinding{"right", right_data},
  }};
  const ReferenceResult source_result = require_reference(
      ReferenceInterpreter::run(source, bindings));
  const ReferenceResult canonical_result = require_reference(
      ReferenceInterpreter::run(canonicalized.graph(), bindings));
  const Tensor* source_output = source_result.output("result");
  const Tensor* result_output = canonical_result.output("result");
  TK_REQUIRE(source_output != nullptr);
  TK_REQUIRE(result_output != nullptr);
  require_bits_equal(source_output->data(), result_output->data());
}

TK_TEST("Structural canonicalization scales across merged sets and many keys") {
  constexpr std::size_t kMergedWidth = 256U;
  constexpr std::size_t kKeyCount = 128U;
  GraphLimits limits;
  limits.max_nodes =
      static_cast<std::uint32_t>(2U + kMergedWidth + 2U * kKeyCount);
  limits.max_outputs = 1U;
  GraphBuilder builder(limits);
  const ValueId left = require_value(builder.input("left", scalar_type()));
  const ValueId right = require_value(builder.input("right", scalar_type()));
  std::vector<ValueId> merged_values;
  merged_values.reserve(kMergedWidth);
  for (std::size_t index = 0U; index < kMergedWidth; ++index) {
    merged_values.push_back(require_value(builder.add(left, right)));
  }
  ValueId current = merged_values[0];
  for (std::size_t index = 0U; index < kKeyCount; ++index) {
    const ValueId first = require_value(builder.add(current, right));
    const ValueId second = require_value(builder.add(current, right));
    TK_REQUIRE(first != second);
    current = first;
  }
  require_output(builder.output("result", current));
  const VerifiedGraph source = require_graph(std::move(builder).finish());

  const StructuralCanonicalizationResult canonicalized =
      require_canonicalized(StructuralCanonicalization::run(source));
  const StructuralCanonicalizationStats expected_stats{
      static_cast<std::uint32_t>(2U + kMergedWidth + 2U * kKeyCount),
      static_cast<std::uint32_t>(3U + kKeyCount),
      static_cast<std::uint32_t>(kMergedWidth - 1U + kKeyCount),
      static_cast<std::uint32_t>(kMergedWidth - 1U + kKeyCount),
      0U,
      0U,
  };
  TK_REQUIRE_EQ(canonicalized.stats(), expected_stats);
  TK_REQUIRE_EQ(canonicalized.graph().limits(), limits);
  const NodeProvenance* merged =
      canonicalized.provenance().for_source(merged_values[0]);
  TK_REQUIRE(merged != nullptr);
  TK_REQUIRE_EQ(merged->sources().size(), kMergedWidth);
  for (const ValueId value : merged_values) {
    TK_REQUIRE(canonicalized.provenance().for_source(value) == merged);
  }

  const std::array<float, 1U> left_data{{0.125F}};
  const std::array<float, 1U> right_data{{-0.0625F}};
  const std::array<InputBinding, 2U> bindings{{
      InputBinding{"left", left_data},
      InputBinding{"right", right_data},
  }};
  const ReferenceResult source_result = require_reference(
      ReferenceInterpreter::run(source, bindings));
  const ReferenceResult canonical_result = require_reference(
      ReferenceInterpreter::run(canonicalized.graph(), bindings));
  const Tensor* source_output = source_result.output("result");
  const Tensor* result_output = canonical_result.output("result");
  TK_REQUIRE(source_output != nullptr);
  TK_REQUIRE(result_output != nullptr);
  require_bits_equal(source_output->data(), result_output->data());
}

TK_TEST("Structural canonicalization scales across protected equivalence classes") {
  constexpr std::size_t kWidth = 512U;
  GraphLimits limits;
  limits.max_nodes = static_cast<std::uint32_t>(kWidth + 1U);
  limits.max_outputs = static_cast<std::uint32_t>(kWidth);
  limits.max_name_bytes = 32U;
  GraphBuilder builder(limits);
  const ValueId input = require_value(builder.input("input", scalar_type()));
  std::vector<ValueId> values;
  values.reserve(kWidth);
  for (std::size_t index = 0U; index < kWidth; ++index) {
    values.push_back(require_value(builder.relu(input)));
  }
  for (std::size_t index = 0U; index < values.size(); ++index) {
    require_output(
        builder.output("result_" + std::to_string(index), values[index]));
  }
  const VerifiedGraph source = require_graph(std::move(builder).finish());

  const StructuralCanonicalizationResult first =
      require_canonicalized(StructuralCanonicalization::run(source));
  const StructuralCanonicalizationResult repeated =
      require_canonicalized(StructuralCanonicalization::run(source));
  const StructuralCanonicalizationStats expected_stats{
      static_cast<std::uint32_t>(kWidth + 1U),
      static_cast<std::uint32_t>(kWidth + 1U),
      0U,
      0U,
      0U,
      static_cast<std::uint32_t>(kWidth - 1U),
  };
  TK_REQUIRE_EQ(first.stats(), expected_stats);
  TK_REQUIRE_EQ(first.graph().limits(), limits);
  TK_REQUIRE_EQ(first.graph().dump(), source.dump());
  TK_REQUIRE_EQ(first.provenance().entries().size(), source.nodes().size());
  TK_REQUIRE_EQ(repeated.stats(), first.stats());
  TK_REQUIRE_EQ(repeated.graph().dump(), first.graph().dump());
  TK_REQUIRE_EQ(repeated.provenance().dump(), first.provenance().dump());
}

}  // namespace
