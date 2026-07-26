#include "test.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
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

  [[nodiscard]] float finite_quarter() noexcept {
    const auto numerator =
        static_cast<std::int32_t>(bounded(65U)) - 32;
    return static_cast<float>(numerator) * 0.25F;
  }

 private:
  std::uint64_t state_;
};

[[nodiscard]] tensorkiln::TensorType f32(
    const std::vector<std::int64_t>& extents) {
  return unwrap(tensorkiln::TensorType::create(
      unwrap(tensorkiln::Shape::create(
          std::span<const std::int64_t>{extents}))));
}

void require_close(const float actual, const float expected) {
  constexpr double absolute_tolerance = 2.0e-6;
  constexpr double relative_tolerance = 2.0e-5;
  const double promoted_actual = static_cast<double>(actual);
  const double promoted_expected = static_cast<double>(expected);
  TK_REQUIRE(std::isfinite(promoted_actual));
  TK_REQUIRE(std::isfinite(promoted_expected));
  TK_REQUIRE(std::abs(promoted_actual - promoted_expected) <=
             absolute_tolerance +
                 relative_tolerance * std::abs(promoted_expected));
}

void require_normalized_last_axis(const std::span<const float> values,
                                  const std::size_t slice_extent) {
  TK_REQUIRE(slice_extent > 0U);
  TK_REQUIRE(values.size() % slice_extent == 0U);
  for (std::size_t base = 0U; base < values.size();
       base += slice_extent) {
    double sum = 0.0;
    for (std::size_t coordinate = 0U; coordinate < slice_extent;
         ++coordinate) {
      const float value = values[base + coordinate];
      TK_REQUIRE(std::isfinite(value));
      TK_REQUIRE(value >= 0.0F);
      TK_REQUIRE(value <= 1.0F);
      sum += static_cast<double>(value);
    }
    TK_REQUIRE(std::abs(sum - 1.0) <= 2.0e-6);
  }
}

[[nodiscard]] std::vector<float> execute_and_compare(
    tensorkiln::ExecutionSession& session,
    const tensorkiln::VerifiedGraph& graph,
    const std::span<const float> input,
    const std::size_t slice_extent) {
  const std::array<tensorkiln::ExecutionInputBinding, 1U>
      execution_bindings{{{"x", input}}};
  const std::array<tensorkiln::InputBinding, 1U>
      reference_bindings{{{"x", input}}};
  TK_REQUIRE(session.bind(execution_bindings).has_value());
  TK_REQUIRE_EQ(session.run(), tensorkiln::ExecutionRunStatus::success);

  const auto result = session.result();
  TK_REQUIRE(result.has_value());
  const auto actual = result->output("probabilities");
  TK_REQUIRE(actual.has_value());
  const tensorkiln::ReferenceResult reference = unwrap(
      tensorkiln::ReferenceInterpreter::run(graph, reference_bindings));
  const tensorkiln::Tensor* expected =
      reference.output("probabilities");
  TK_REQUIRE(expected != nullptr);
  TK_REQUIRE_EQ(actual->type(), expected->type());
  TK_REQUIRE_EQ(actual->data().size(), expected->data().size());
  for (std::size_t index = 0U; index < actual->data().size(); ++index) {
    require_close(actual->data()[index], expected->data()[index]);
  }
  require_normalized_last_axis(actual->data(), slice_extent);
  return std::vector<float>{actual->data().begin(), actual->data().end()};
}

TK_TEST("Last-axis Softmax is differential normalized and translation invariant") {
  constexpr std::size_t seed_count = 96U;
  std::array<std::size_t, 4U> rank_coverage{};
  std::array<std::size_t, 7U> extent_coverage{};

  for (std::size_t index = 0U; index < seed_count; ++index) {
    const std::uint64_t seed =
        UINT64_C(0x736f66746d61786b) +
        static_cast<std::uint64_t>(index);
    SplitMix64 random(seed);
    const std::size_t rank = index % 4U + 1U;
    std::vector<std::int64_t> extents(rank);
    for (std::size_t axis = 0U; axis + 1U < rank; ++axis) {
      extents[axis] =
          static_cast<std::int64_t>(random.bounded(3U) + 1U);
    }
    const std::size_t slice_extent =
        static_cast<std::size_t>(random.bounded(7U) + 1U);
    extents.back() = static_cast<std::int64_t>(slice_extent);
    ++rank_coverage[rank - 1U];
    ++extent_coverage[slice_extent - 1U];

    const tensorkiln::TensorType type = f32(extents);
    std::vector<float> input;
    input.reserve(static_cast<std::size_t>(type.numel()));
    for (std::uint64_t element = 0U; element < type.numel(); ++element) {
      input.push_back(random.finite_quarter());
    }

    tensorkiln::GraphBuilder builder;
    const tensorkiln::ValueId input_value =
        unwrap(builder.input("x", type));
    const tensorkiln::ValueId probabilities =
        unwrap(builder.softmax(input_value, -1));
    static_cast<void>(
        unwrap(builder.output("probabilities", probabilities)));
    const tensorkiln::VerifiedGraph graph =
        unwrap(std::move(builder).finish());
    const tensorkiln::ExecutionPlan plan =
        unwrap(tensorkiln::ExecutionPlanCompiler::run(graph));
    TK_REQUIRE_EQ(plan.steps().size(), 1U);
    TK_REQUIRE_EQ(
        plan.steps()[0].kernel(),
        tensorkiln::DenseKernelKind::softmax_last_axis_f32);
    TK_REQUIRE_EQ(plan.steps()[0].scalar_steps(), 3U * type.numel());

    tensorkiln::ExecutionSession session =
        tensorkiln::ExecutionSession::create(
            plan, tensorkiln::ExecutionSessionOptions{true});
    TK_REQUIRE(session.audits_kernel_writes());
    const std::vector<float> first =
        execute_and_compare(session, graph, input, slice_extent);
    const std::vector<float> repeated =
        execute_and_compare(session, graph, input, slice_extent);
    TK_REQUIRE_EQ(first.size(), repeated.size());
    for (std::size_t element = 0U; element < first.size(); ++element) {
      require_close(repeated[element], first[element]);
    }

    std::vector<float> translated;
    translated.reserve(input.size());
    for (const float value : input) {
      translated.push_back(value + 128.0F);
    }
    const std::vector<float> shifted =
        execute_and_compare(session, graph, translated, slice_extent);
    TK_REQUIRE_EQ(first.size(), shifted.size());
    for (std::size_t element = 0U; element < first.size(); ++element) {
      require_close(shifted[element], first[element]);
    }
  }

  for (const std::size_t covered : rank_coverage) {
    TK_REQUIRE(covered > 0U);
  }
  for (const std::size_t covered : extent_coverage) {
    TK_REQUIRE(covered > 0U);
  }
}

}  // namespace
