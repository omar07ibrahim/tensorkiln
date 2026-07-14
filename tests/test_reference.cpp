#include "test.hpp"

#include <array>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include "oracle_fixture.hpp"
#include "tensorkiln/reference.hpp"

namespace {

using tensorkiln::ErrorCode;
using tensorkiln::GraphBuilder;
using tensorkiln::InputBinding;
using tensorkiln::OutputId;
using tensorkiln::ReferenceInterpreter;
using tensorkiln::ReferenceLimits;
using tensorkiln::ReferenceResult;
using tensorkiln::Shape;
using tensorkiln::Tensor;
using tensorkiln::TensorType;
using tensorkiln::ValueId;
using tensorkiln::VerifiedGraph;

[[nodiscard]] TensorType make_type(
    const std::span<const std::int64_t> extents) {
  const auto shape = Shape::create(extents);
  TK_REQUIRE(shape.value_if() != nullptr);
  const auto type = TensorType::create(*shape.value_if());
  TK_REQUIRE(type.value_if() != nullptr);
  return *type.value_if();
}

[[nodiscard]] TensorType make_type(
    const std::initializer_list<std::int64_t> extents) {
  return make_type(std::span<const std::int64_t>(extents.begin(),
                                                 extents.size()));
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

[[nodiscard]] ReferenceResult require_result(
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

void require_bits_equal(const std::span<const float> actual,
                        const std::span<const float> expected) {
  TK_REQUIRE_EQ(actual.size(), expected.size());
  for (std::size_t index = 0U; index < actual.size(); ++index) {
    TK_REQUIRE_EQ(std::bit_cast<std::uint32_t>(actual[index]),
                  std::bit_cast<std::uint32_t>(expected[index]));
  }
}

[[nodiscard]] const Tensor& require_output_tensor(
    const ReferenceResult& result, const std::string_view name) {
  const Tensor* output = result.output(name);
  TK_REQUIRE(output != nullptr);
  return *output;
}

class RoundingGuard final {
 public:
  RoundingGuard() : original_(std::fegetround()) {}
  RoundingGuard(const RoundingGuard&) = delete;
  RoundingGuard& operator=(const RoundingGuard&) = delete;
  ~RoundingGuard() { static_cast<void>(std::fesetround(original_)); }

 private:
  int original_;
};

TK_TEST("Reference binding diagnostics have phase-stable precedence") {
  GraphBuilder builder;
  const ValueId left = require_value(builder.input("left", make_type({2})));
  const ValueId right = require_value(builder.input("right", make_type({2})));
  const ValueId auxiliary =
      require_value(builder.input("auxiliary", make_type({2})));
  static_cast<void>(auxiliary);
  const ValueId sum = require_value(builder.add(left, right));
  require_output(builder.output("result", sum));
  const VerifiedGraph graph = require_graph(std::move(builder).finish());

  const std::array<float, 1U> short_data{{1.0F}};
  const std::array<float, 2U> exact_data{{1.0F, 2.0F}};
  const std::array<InputBinding, 3U> unknown_and_duplicate{{
      InputBinding{"left", short_data},
      InputBinding{"left", short_data},
      InputBinding{"ghost", exact_data},
  }};
  require_error(ReferenceInterpreter::run(graph, unknown_and_duplicate),
                ErrorCode::input_binding_unknown);

  const std::array<InputBinding, 4U> excessive{{
      InputBinding{"left", exact_data},
      InputBinding{"right", exact_data},
      InputBinding{"auxiliary", exact_data},
      InputBinding{"ghost", exact_data},
  }};
  require_error(ReferenceInterpreter::run(graph, excessive),
                ErrorCode::input_binding_count_exceeded);

  const std::array<InputBinding, 2U> duplicate{{
      InputBinding{"left", short_data},
      InputBinding{"left", exact_data},
  }};
  require_error(ReferenceInterpreter::run(graph, duplicate),
                ErrorCode::input_binding_duplicate);

  const std::array<InputBinding, 1U> missing{{
      InputBinding{"left", short_data},
  }};
  require_error(ReferenceInterpreter::run(graph, missing),
                ErrorCode::input_binding_missing);

  const std::array<InputBinding, 2U> wrong_size{{
      InputBinding{"left", short_data},
      InputBinding{"right", exact_data},
  }};
  const std::array<InputBinding, 3U> complete_wrong_size{{
      wrong_size[0],
      wrong_size[1],
      InputBinding{"auxiliary", exact_data},
  }};
  const auto result = ReferenceInterpreter::run(graph, complete_wrong_size);
  const auto& diagnostic =
      require_error(result, ErrorCode::input_binding_size_mismatch);
  TK_REQUIRE_EQ(diagnostic.message,
                "input left expects 2 f32 values; received 1");
}

TK_TEST("Reference results own feeds and reject foreign value IDs") {
  GraphBuilder builder;
  const ValueId input = require_value(builder.input("x", make_type({3})));
  require_output(builder.output("first", input));
  require_output(builder.output("alias", input));
  const VerifiedGraph graph = require_graph(std::move(builder).finish());

  const float payload_nan = std::bit_cast<float>(0x7fc12345U);
  std::array<float, 3U> payload{{-0.0F, payload_nan, 9.0F}};
  const std::array<InputBinding, 1U> bindings{{InputBinding{"x", payload}}};
  const ReferenceResult result =
      require_result(ReferenceInterpreter::run(graph, bindings));
  payload.fill(0.0F);

  const Tensor* by_value = result.value(input);
  TK_REQUIRE(by_value != nullptr);
  TK_REQUIRE(result.output("first") == by_value);
  TK_REQUIRE(result.output("alias") == by_value);
  TK_REQUIRE(result.output("missing") == nullptr);
  TK_REQUIRE_EQ(std::bit_cast<std::uint32_t>(by_value->data()[0]),
                0x80000000U);
  TK_REQUIRE_EQ(std::bit_cast<std::uint32_t>(by_value->data()[1]),
                0x7fc12345U);
  TK_REQUIRE_EQ(by_value->data()[2], 9.0F);

  GraphBuilder foreign_builder;
  const ValueId foreign =
      require_value(foreign_builder.input("x", make_type({3})));
  require_output(foreign_builder.output("result", foreign));
  const VerifiedGraph foreign_graph =
      require_graph(std::move(foreign_builder).finish());
  static_cast<void>(foreign_graph);
  TK_REQUIRE(result.value(foreign) == nullptr);
}

TK_TEST("Reference Add broadcasts trailing axes in both operand orders") {
  GraphBuilder builder;
  const ValueId left =
      require_value(builder.input("left", make_type({2, 1, 3, 1})));
  const ValueId right =
      require_value(builder.input("right", make_type({1, 4, 1, 5})));
  const ValueId forward = require_value(builder.add(left, right));
  const ValueId reverse = require_value(builder.add(right, left));
  require_output(builder.output("forward", forward));
  require_output(builder.output("reverse", reverse));
  const VerifiedGraph graph = require_graph(std::move(builder).finish());

  std::vector<float> left_data;
  for (std::size_t batch = 0U; batch < 2U; ++batch) {
    for (std::size_t channel = 0U; channel < 3U; ++channel) {
      left_data.push_back(
          static_cast<float>(100U * batch + 10U * channel));
    }
  }
  std::vector<float> right_data;
  for (std::size_t group = 0U; group < 4U; ++group) {
    for (std::size_t column = 0U; column < 5U; ++column) {
      right_data.push_back(static_cast<float>(1000U * group + column));
    }
  }
  const std::array<InputBinding, 2U> bindings{{
      InputBinding{"left", left_data},
      InputBinding{"right", right_data},
  }};
  const ReferenceResult result =
      require_result(ReferenceInterpreter::run(graph, bindings));

  std::vector<float> expected;
  for (std::size_t batch = 0U; batch < 2U; ++batch) {
    for (std::size_t group = 0U; group < 4U; ++group) {
      for (std::size_t channel = 0U; channel < 3U; ++channel) {
        for (std::size_t column = 0U; column < 5U; ++column) {
          expected.push_back(static_cast<float>(100U * batch +
                                                1000U * group +
                                                10U * channel + column));
        }
      }
    }
  }
  require_bits_equal(require_output_tensor(result, "forward").data(),
                     expected);
  require_bits_equal(require_output_tensor(result, "reverse").data(),
                     expected);
}

TK_TEST("Reference Add right-aligns operands with different ranks") {
  GraphBuilder builder;
  const ValueId left =
      require_value(builder.input("left", make_type({2, 1, 3})));
  const ValueId right = require_value(builder.input("right", make_type({2, 1})));
  const ValueId forward = require_value(builder.add(left, right));
  const ValueId reverse = require_value(builder.add(right, left));
  require_output(builder.output("forward", forward));
  require_output(builder.output("reverse", reverse));
  const VerifiedGraph graph = require_graph(std::move(builder).finish());

  const std::array<float, 6U> left_data{{1.0F, 2.0F, 3.0F,
                                          10.0F, 20.0F, 30.0F}};
  const std::array<float, 2U> right_data{{100.0F, 1000.0F}};
  const std::array<float, 12U> expected{{
      101.0F, 102.0F, 103.0F, 1001.0F, 1002.0F, 1003.0F,
      110.0F, 120.0F, 130.0F, 1010.0F, 1020.0F, 1030.0F,
  }};
  const std::array<InputBinding, 2U> bindings{{
      InputBinding{"left", left_data},
      InputBinding{"right", right_data},
  }};
  const ReferenceResult result =
      require_result(ReferenceInterpreter::run(graph, bindings));

  require_bits_equal(require_output_tensor(result, "forward").data(),
                     expected);
  require_bits_equal(require_output_tensor(result, "reverse").data(),
                     expected);
}

TK_TEST("Reference MatMul broadcasts independent rank-four batches") {
  GraphBuilder builder;
  const ValueId left =
      require_value(builder.input("left", make_type({2, 1, 2, 3})));
  const ValueId right =
      require_value(builder.input("right", make_type({1, 3, 3, 2})));
  const ValueId product = require_value(builder.matmul(left, right));
  require_output(builder.output("result", product));
  const VerifiedGraph graph = require_graph(std::move(builder).finish());

  const std::array<float, 12U> left_data{{
      1.0F,  2.0F,  3.0F,  4.0F,  5.0F,  6.0F,
      7.0F,  8.0F,  9.0F,  10.0F, 11.0F, 12.0F,
  }};
  const std::array<float, 18U> right_data{{
      1.0F,  0.0F, 0.0F,  1.0F, 1.0F, 1.0F,
      2.0F,  1.0F, 1.0F,  0.0F, 0.0F, 2.0F,
      -1.0F, 1.0F, 2.0F, -1.0F, 1.0F, 0.0F,
  }};
  const std::array<float, 24U> expected{{
      4.0F,  5.0F,  10.0F, 11.0F, 4.0F,  7.0F,
      13.0F, 16.0F, 6.0F,  -1.0F, 12.0F, -1.0F,
      16.0F, 17.0F, 22.0F, 23.0F, 22.0F, 25.0F,
      31.0F, 34.0F, 18.0F, -1.0F, 24.0F, -1.0F,
  }};
  const std::array<InputBinding, 2U> bindings{{
      InputBinding{"left", left_data},
      InputBinding{"right", right_data},
  }};
  const ReferenceResult result =
      require_result(ReferenceInterpreter::run(graph, bindings));

  const Tensor& output = require_output_tensor(result, "result");
  TK_REQUIRE_EQ(output.type().shape().to_string(), "[2,3,2,2]");
  require_bits_equal(output.data(), expected);
}

TK_TEST("Reference MatMul right-aligns unequal batch ranks") {
  GraphBuilder builder;
  const ValueId left =
      require_value(builder.input("left", make_type({2, 1, 2})));
  const ValueId right =
      require_value(builder.input("right", make_type({2, 1, 2, 1})));
  const ValueId product = require_value(builder.matmul(left, right));
  require_output(builder.output("result", product));
  const VerifiedGraph graph = require_graph(std::move(builder).finish());

  const std::array<float, 4U> left_data{{1.0F, 2.0F, 3.0F, 4.0F}};
  const std::array<float, 4U> right_data{{10.0F, 20.0F, -1.0F, 2.0F}};
  const std::array<float, 4U> expected{{50.0F, 110.0F, 3.0F, 5.0F}};
  const std::array<InputBinding, 2U> bindings{{
      InputBinding{"left", left_data},
      InputBinding{"right", right_data},
  }};
  const ReferenceResult result =
      require_result(ReferenceInterpreter::run(graph, bindings));

  const Tensor& output = require_output_tensor(result, "result");
  TK_REQUIRE_EQ(output.type().shape().to_string(), "[2,2,1,1]");
  require_bits_equal(output.data(), expected);
  TK_REQUIRE_EQ(result.materialized_bytes(), 48U);
  TK_REQUIRE_EQ(result.scalar_steps(), 16U);
}

TK_TEST("Reference MatMul keeps products and accumulation in double") {
  GraphBuilder builder;
  const ValueId left =
      require_value(builder.input("left", make_type({1, 5})));
  const ValueId right =
      require_value(builder.input("right", make_type({5, 2})));
  const ValueId product = require_value(builder.matmul(left, right));
  require_output(builder.output("result", product));
  const VerifiedGraph graph = require_graph(std::move(builder).finish());

  const float near_one = std::bit_cast<float>(0x3f7ffff8U);
  const float adjacent = std::bit_cast<float>(0x3f7ffff9U);
  const std::array<float, 5U> left_data{{
      16777216.0F, 1.0F, -16777216.0F, near_one, near_one,
  }};
  const std::array<float, 10U> right_data{{
      1.0F, 0.0F, 1.0F, 0.0F, 1.0F,
      0.0F, 0.0F, near_one, 0.0F, adjacent,
  }};
  const std::array<InputBinding, 2U> bindings{{
      InputBinding{"left", left_data},
      InputBinding{"right", right_data},
  }};
  const ReferenceResult result =
      require_result(ReferenceInterpreter::run(graph, bindings));

  const std::span<const float> output =
      require_output_tensor(result, "result").data();
  TK_REQUIRE_EQ(output.size(), 2U);
  TK_REQUIRE_EQ(std::bit_cast<std::uint32_t>(output[0]), 0x3f800000U);
  TK_REQUIRE_EQ(std::bit_cast<std::uint32_t>(output[1]), 0x3ffffff1U);
  TK_REQUIRE_EQ(result.materialized_bytes(), 68U);
  TK_REQUIRE_EQ(result.scalar_steps(), 25U);
}

TK_TEST("Reference ReLU defines NaN infinity and signed-zero behavior") {
  GraphBuilder builder;
  const ValueId input = require_value(builder.input("x", make_type({7})));
  const ValueId activated = require_value(builder.relu(input));
  require_output(builder.output("result", activated));
  const VerifiedGraph graph = require_graph(std::move(builder).finish());

  const std::array<float, 7U> data{{
      -std::numeric_limits<float>::infinity(),
      std::bit_cast<float>(0x80000001U),
      -0.0F,
      0.0F,
      std::bit_cast<float>(0x00000001U),
      std::numeric_limits<float>::infinity(),
      std::bit_cast<float>(0x7fc12345U),
  }};
  const std::array<InputBinding, 1U> bindings{{InputBinding{"x", data}}};
  const ReferenceResult result =
      require_result(ReferenceInterpreter::run(graph, bindings));
  const std::span<const float> output =
      require_output_tensor(result, "result").data();

  for (std::size_t index = 0U; index < 4U; ++index) {
    TK_REQUIRE_EQ(std::bit_cast<std::uint32_t>(output[index]), 0x00000000U);
  }
  TK_REQUIRE_EQ(std::bit_cast<std::uint32_t>(output[4]), 0x00000001U);
  TK_REQUIRE_EQ(std::bit_cast<std::uint32_t>(output[5]), 0x7f800000U);
  TK_REQUIRE(std::isnan(output[6]));
  TK_REQUIRE_EQ(std::bit_cast<std::uint32_t>(output[6]), 0x7fc12345U);
}

TK_TEST("Reference limits accept exact payload and scalar-step boundaries") {
  GraphBuilder builder;
  const ValueId input = require_value(builder.input("x", make_type({2})));
  const ValueId activated = require_value(builder.relu(input));
  require_output(builder.output("result", activated));
  require_output(builder.output("alias", activated));
  const VerifiedGraph graph = require_graph(std::move(builder).finish());

  const std::array<float, 2U> data{{-1.0F, 2.0F}};
  const std::array<InputBinding, 1U> bindings{{InputBinding{"x", data}}};
  const ReferenceLimits exact{16U, 4U};
  const ReferenceResult result =
      require_result(ReferenceInterpreter::run(graph, bindings, exact));
  TK_REQUIRE_EQ(result.materialized_bytes(), 16U);
  TK_REQUIRE_EQ(result.scalar_steps(), 4U);
  TK_REQUIRE(result.output("result") == result.output("alias"));

  require_error(ReferenceInterpreter::run(
                    graph, bindings, ReferenceLimits{15U, 4U}),
                ErrorCode::reference_materialization_limit_exceeded);
  require_error(ReferenceInterpreter::run(
                    graph, bindings, ReferenceLimits{16U, 3U}),
                ErrorCode::reference_scalar_step_limit_exceeded);
  require_error(ReferenceInterpreter::run(
                    graph, bindings, ReferenceLimits{15U, 0U}),
                ErrorCode::reference_materialization_limit_exceeded);
}

TK_TEST("Reference interpreter rejects non-nearest rounding without mutation") {
  GraphBuilder builder;
  const std::array<float, 1U> constant_data{{1.0F}};
  const ValueId constant = require_value(
      builder.constant("one", make_type({1}), constant_data));
  require_output(builder.output("result", constant));
  const VerifiedGraph graph = require_graph(std::move(builder).finish());

  const RoundingGuard guard;
  TK_REQUIRE_EQ(std::fesetround(FE_DOWNWARD), 0);
  require_error(ReferenceInterpreter::run(graph, {}),
                ErrorCode::unsupported_rounding_mode);
  TK_REQUIRE_EQ(std::fegetround(), FE_DOWNWARD);
}

TK_TEST("Reference MLP matches every independent oracle boundary") {
  using namespace tensorkiln::oracle_fixture;

  GraphBuilder builder;
  const ValueId input =
      require_value(builder.input("x", make_type(kMlpInputShape)));
  const ValueId weights = require_value(
      builder.constant("weights", make_type(kMlpWeightShape), kMlpWeights));
  const ValueId bias = require_value(
      builder.constant("bias", make_type(kMlpBiasShape), kMlpBias));
  const ValueId product = require_value(builder.matmul(input, weights));
  const ValueId pre_activation = require_value(builder.add(product, bias));
  const ValueId activated = require_value(builder.relu(pre_activation));
  require_output(builder.output("product", product));
  require_output(builder.output("pre_activation", pre_activation));
  require_output(builder.output("result", activated));
  const VerifiedGraph graph = require_graph(std::move(builder).finish());

  const std::array<InputBinding, 1U> bindings{{
      InputBinding{"x", kMlpInput},
  }};
  const ReferenceResult result =
      require_result(ReferenceInterpreter::run(graph, bindings));

  require_bits_equal(require_output_tensor(result, "product").data(),
                     kMlpProduct);
  require_bits_equal(
      require_output_tensor(result, "pre_activation").data(),
      kMlpPreActivation);
  require_bits_equal(require_output_tensor(result, "result").data(),
                     kMlpOutput);
  TK_REQUIRE_EQ(require_output_tensor(result, "result").type().shape(),
                make_type(kMlpOutputShape).shape());
  TK_REQUIRE_EQ(result.materialized_bytes(), 184U);
  TK_REQUIRE_EQ(result.scalar_steps(), 62U);
}

}  // namespace
