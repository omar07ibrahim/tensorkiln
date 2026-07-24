#include "test.hpp"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "tensorkiln/execution.hpp"
#include "tensorkiln/reference.hpp"

namespace {

template <typename T>
[[nodiscard]] T unwrap(tensorkiln::Result<T> result) {
  TK_REQUIRE(result.value_if() != nullptr);
  return std::move(*result.value_if());
}

class SplitMix64 final {
 public:
  explicit SplitMix64(const std::uint64_t seed) : state_(seed) {}

  [[nodiscard]] std::uint64_t next() noexcept {
    state_ += UINT64_C(0x9e3779b97f4a7c15);
    std::uint64_t value = state_;
    value = (value ^ (value >> 30U)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27U)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31U);
  }

  [[nodiscard]] std::uint64_t bounded(const std::uint64_t limit) noexcept {
    TK_REQUIRE(limit > 0U);
    return next() % limit;
  }

  [[nodiscard]] float finite_value() noexcept {
    const auto numerator = static_cast<std::int32_t>(bounded(33U)) - 16;
    return static_cast<float>(numerator) * 0.125F;
  }

 private:
  std::uint64_t state_;
};

[[nodiscard]] tensorkiln::TensorType f32(
    const std::vector<std::int64_t>& extents) {
  return unwrap(tensorkiln::TensorType::create(unwrap(
      tensorkiln::Shape::create(std::span<const std::int64_t>{extents}))));
}

struct GeneratedInput final {
  std::string name;
  std::vector<float> first;
  std::vector<float> second;
};

struct GeneratedFixture final {
  tensorkiln::VerifiedGraph graph;
  std::vector<GeneratedInput> inputs;
  std::size_t mode;
};

[[nodiscard]] tensorkiln::ValueId add_input(
    tensorkiln::GraphBuilder& builder, std::vector<GeneratedInput>& inputs,
    std::string name, const tensorkiln::TensorType type, SplitMix64& random) {
  const tensorkiln::ValueId value = unwrap(builder.input(name, type));
  const std::size_t elements = static_cast<std::size_t>(type.numel());
  GeneratedInput generated{std::move(name), {}, {}};
  generated.first.reserve(elements);
  generated.second.reserve(elements);
  for (std::size_t index = 0U; index < elements; ++index) {
    generated.first.push_back(random.finite_value());
    generated.second.push_back(random.finite_value());
  }
  inputs.push_back(std::move(generated));
  return value;
}

[[nodiscard]] tensorkiln::ValueId add_random_constant(
    tensorkiln::GraphBuilder& builder, const std::string_view name,
    const tensorkiln::TensorType type, SplitMix64& random) {
  std::vector<float> data;
  data.reserve(static_cast<std::size_t>(type.numel()));
  for (std::uint64_t index = 0U; index < type.numel(); ++index) {
    data.push_back(random.finite_value());
  }
  return unwrap(builder.constant(std::string(name), type,
                                 std::span<const float>{data}));
}

void add_common_outputs(tensorkiln::GraphBuilder& builder,
                        const tensorkiln::ValueId early,
                        const tensorkiln::ValueId result) {
  static_cast<void>(unwrap(builder.output("early", early)));
  static_cast<void>(unwrap(builder.output("result", result)));
  static_cast<void>(unwrap(builder.output("result_alias", result)));
}

[[nodiscard]] GeneratedFixture make_fixture(const std::uint64_t seed) {
  SplitMix64 random(seed);
  tensorkiln::GraphBuilder builder;
  std::vector<GeneratedInput> inputs;
  const std::size_t mode = static_cast<std::size_t>(seed % 4U);

  if (mode == 0U) {
    const std::int64_t rows =
        static_cast<std::int64_t>(random.bounded(3U) + 1U);
    const std::int64_t columns =
        static_cast<std::int64_t>(random.bounded(4U) + 1U);
    const tensorkiln::ValueId value =
        add_input(builder, inputs, "x", f32({rows, columns}), random);
    const tensorkiln::ValueId bias =
        add_input(builder, inputs, "bias", f32({columns}), random);
    const tensorkiln::ValueId sum = unwrap(builder.add(value, bias));
    const tensorkiln::ValueId early = unwrap(builder.relu(sum));
    const tensorkiln::ValueId doubled = unwrap(builder.add(early, early));
    const tensorkiln::ValueId result = unwrap(builder.relu(doubled));
    static_cast<void>(unwrap(builder.add(value, bias)));
    add_common_outputs(builder, early, result);
  } else if (mode == 1U) {
    const std::int64_t rows =
        static_cast<std::int64_t>(random.bounded(3U) + 1U);
    const std::int64_t inner =
        static_cast<std::int64_t>(random.bounded(4U) + 1U);
    const std::int64_t columns =
        static_cast<std::int64_t>(random.bounded(3U) + 1U);
    const tensorkiln::ValueId left =
        add_input(builder, inputs, "left", f32({rows, inner}), random);
    const tensorkiln::ValueId right =
        add_input(builder, inputs, "right", f32({inner, columns}), random);
    const tensorkiln::ValueId product = unwrap(builder.matmul(left, right));
    const tensorkiln::ValueId bias = add_random_constant(
        builder, "bias", f32({columns}), random);
    const tensorkiln::ValueId shifted = unwrap(builder.add(product, bias));
    const tensorkiln::ValueId early = unwrap(builder.relu(shifted));
    const tensorkiln::ValueId result = unwrap(builder.add(early, early));
    add_common_outputs(builder, product, result);
  } else if (mode == 2U) {
    const std::int64_t batches =
        static_cast<std::int64_t>(random.bounded(2U) + 1U);
    const std::int64_t channels =
        static_cast<std::int64_t>(random.bounded(3U) + 1U);
    const std::int64_t rows =
        static_cast<std::int64_t>(random.bounded(2U) + 1U);
    const std::int64_t inner =
        static_cast<std::int64_t>(random.bounded(3U) + 1U);
    const std::int64_t columns =
        static_cast<std::int64_t>(random.bounded(3U) + 1U);
    const tensorkiln::ValueId left = add_input(
        builder, inputs, "left", f32({batches, 1, rows, inner}), random);
    const tensorkiln::ValueId right = add_input(
        builder, inputs, "right", f32({1, channels, inner, columns}), random);
    const tensorkiln::ValueId product = unwrap(builder.matmul(left, right));
    const tensorkiln::ValueId bias = add_random_constant(
        builder, "bias", f32({columns}), random);
    const tensorkiln::ValueId shifted = unwrap(builder.add(product, bias));
    const tensorkiln::ValueId early = unwrap(builder.relu(shifted));
    const tensorkiln::ValueId result = unwrap(builder.add(early, early));
    add_common_outputs(builder, product, result);
  } else {
    const tensorkiln::ValueId left =
        add_input(builder, inputs, "left", f32({}), random);
    const tensorkiln::ValueId right =
        add_input(builder, inputs, "right", f32({}), random);
    const tensorkiln::ValueId sum = unwrap(builder.add(left, right));
    const tensorkiln::ValueId early = unwrap(builder.relu(sum));
    const tensorkiln::ValueId shifted = unwrap(builder.add(early, left));
    const tensorkiln::ValueId result = unwrap(builder.relu(shifted));
    static_cast<void>(unwrap(builder.add(left, right)));
    add_common_outputs(builder, early, result);
  }

  return GeneratedFixture{
      unwrap(std::move(builder).finish()),
      std::move(inputs),
      mode,
  };
}

using OutputBits = std::vector<std::vector<std::uint32_t>>;

[[nodiscard]] OutputBits run_and_compare(
    tensorkiln::ExecutionSession& session, const GeneratedFixture& fixture,
    const bool second, const std::uint64_t seed) {
  std::vector<tensorkiln::ExecutionInputBinding> execution_bindings;
  std::vector<tensorkiln::InputBinding> reference_bindings;
  execution_bindings.reserve(fixture.inputs.size());
  reference_bindings.reserve(fixture.inputs.size());
  for (const GeneratedInput& input : fixture.inputs) {
    const std::span<const float> data =
        second ? std::span<const float>{input.second}
               : std::span<const float>{input.first};
    execution_bindings.push_back({input.name, data});
    reference_bindings.push_back({input.name, data});
  }

  const auto bound = session.bind(execution_bindings);
  if (!bound.has_value()) {
    throw tensorkiln::test::Failure(
        "seed " + std::to_string(seed) + " failed execution binding: " +
        std::string(tensorkiln::error_code_name(bound.error_if()->code)));
  }
  if (session.run() != tensorkiln::ExecutionRunStatus::success) {
    throw tensorkiln::test::Failure("seed " + std::to_string(seed) +
                                    " failed execution run");
  }
  const auto result = session.result();
  TK_REQUIRE(result.has_value());
  const tensorkiln::ReferenceResult reference = unwrap(
      tensorkiln::ReferenceInterpreter::run(fixture.graph,
                                             reference_bindings));

  OutputBits captured;
  captured.reserve(fixture.graph.outputs().size());
  for (const tensorkiln::GraphOutput& output : fixture.graph.outputs()) {
    const auto actual = result->output(output.name());
    const tensorkiln::Tensor* expected = reference.output(output.name());
    if (!actual.has_value() || expected == nullptr ||
        actual->type() != expected->type() ||
        actual->data().size() != expected->data().size()) {
      throw tensorkiln::test::Failure(
          "seed " + std::to_string(seed) + " changed output contract " +
          output.name());
    }

    std::vector<std::uint32_t> bits;
    bits.reserve(actual->data().size());
    for (std::size_t index = 0U; index < actual->data().size(); ++index) {
      const std::uint32_t actual_bits =
          std::bit_cast<std::uint32_t>(actual->data()[index]);
      const std::uint32_t expected_bits =
          std::bit_cast<std::uint32_t>(expected->data()[index]);
      if (actual_bits != expected_bits) {
        throw tensorkiln::test::Failure(
            "seed " + std::to_string(seed) + " diverged at output " +
            output.name() + " element " + std::to_string(index));
      }
      bits.push_back(actual_bits);
    }
    captured.push_back(std::move(bits));
  }

  const auto result_output = result->output("result");
  const auto alias_output = result->output("result_alias");
  TK_REQUIRE(result_output.has_value());
  TK_REQUIRE(alias_output.has_value());
  TK_REQUIRE(result_output->data().data() == alias_output->data().data());
  return captured;
}

TK_TEST("Execution is differential and replayable across seeded dense DAGs") {
  std::size_t mode_coverage[4U]{};
  std::size_t kernel_coverage[5U]{};
  std::size_t reused_plans = 0U;
  constexpr std::size_t seed_count = 128U;

  for (std::size_t index = 0U; index < seed_count; ++index) {
    const std::uint64_t seed =
        UINT64_C(0x74656e736f726b69) + static_cast<std::uint64_t>(index);
    const GeneratedFixture first = make_fixture(seed);
    const GeneratedFixture replay = make_fixture(seed);
    TK_REQUIRE_EQ(first.graph.dump(), replay.graph.dump());
    ++mode_coverage[first.mode];

    const tensorkiln::ExecutionPlan first_plan =
        unwrap(tensorkiln::ExecutionPlanCompiler::run(first.graph));
    const tensorkiln::ExecutionPlan replay_plan =
        unwrap(tensorkiln::ExecutionPlanCompiler::run(replay.graph));
    TK_REQUIRE_EQ(first_plan.dump(), replay_plan.dump());
    if (first_plan.arena_plan().stats().total_reserved_bytes >
        first_plan.stats().workspace_bytes) {
      ++reused_plans;
    }
    for (const tensorkiln::ExecutionStep& step : first_plan.steps()) {
      const std::size_t ordinal = static_cast<std::size_t>(step.kernel());
      TK_REQUIRE(ordinal < 5U);
      ++kernel_coverage[ordinal];
    }

    tensorkiln::ExecutionSession session =
        tensorkiln::ExecutionSession::create(
            first_plan, tensorkiln::ExecutionSessionOptions{true});
    TK_REQUIRE(session.audits_kernel_writes());
    const OutputBits first_outputs =
        run_and_compare(session, first, false, seed);
    static_cast<void>(run_and_compare(session, first, true, seed));
    const OutputBits repeated_outputs =
        run_and_compare(session, first, false, seed);
    TK_REQUIRE_EQ(first_outputs, repeated_outputs);
  }

  for (const std::size_t covered : mode_coverage) {
    TK_REQUIRE(covered > 0U);
  }
  for (const std::size_t covered : kernel_coverage) {
    TK_REQUIRE(covered > 0U);
  }
  TK_REQUIRE(reused_plans > seed_count / 2U);
}

}  // namespace
