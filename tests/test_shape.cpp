#include "test.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <vector>

#include "tensorkiln/shape.hpp"

namespace {

using tensorkiln::ErrorCode;
using tensorkiln::Shape;
using tensorkiln::ShapeLimits;

[[nodiscard]] Shape require_shape(
    const tensorkiln::Result<Shape>& result) {
  TK_REQUIRE(result.value_if() != nullptr);
  return *result.value_if();
}

const tensorkiln::Diagnostic& require_error(
    const tensorkiln::Result<Shape>& result, const ErrorCode code) {
  TK_REQUIRE(result.error_if() != nullptr);
  TK_REQUIRE_EQ(result.error_if()->code, code);
  return *result.error_if();
}

TK_TEST("A rank-zero shape is a scalar") {
  const auto result = Shape::create({});
  const Shape shape = require_shape(result);

  TK_REQUIRE(shape.is_scalar());
  TK_REQUIRE_EQ(shape.rank(), 0U);
  TK_REQUIRE_EQ(shape.numel(), 1U);
  TK_REQUIRE(shape.extents().empty());
  TK_REQUIRE(!shape.extent(0U).has_value());
  TK_REQUIRE_EQ(shape.to_string(), "[]");
  TK_REQUIRE_EQ(shape, Shape::scalar());
}

TK_TEST("Shape preserves dimensions and computes element count") {
  const auto result = Shape::create({2, 3, 4});
  const Shape shape = require_shape(result);

  TK_REQUIRE_EQ(shape.rank(), 3U);
  TK_REQUIRE_EQ(shape.numel(), 24U);
  TK_REQUIRE_EQ(shape.extent(0U), std::optional<std::int64_t>(2));
  TK_REQUIRE_EQ(shape.extent(2U), std::optional<std::int64_t>(4));
  TK_REQUIRE(!shape.extent(3U).has_value());
  TK_REQUIRE_EQ(shape.to_string(), "[2,3,4]");
}

TK_TEST("Rank four is accepted at the boundary") {
  const auto result = Shape::create({1, 2, 3, 4});

  TK_REQUIRE(result.has_value());
  TK_REQUIRE_EQ(require_shape(result).rank(), tensorkiln::kMaxRank);
}

TK_TEST("Rank five is rejected before extent validation") {
  const auto result = Shape::create({1, 1, 0, 1, 1});
  const auto& error =
      require_error(result, ErrorCode::rank_limit_exceeded);

  TK_REQUIRE_EQ(error.message, "shape rank 5 exceeds limit 4");
}

TK_TEST("Zero extent reports its axis") {
  const auto result = Shape::create({2, 0, 3});
  const auto& error = require_error(result, ErrorCode::extent_not_positive);

  TK_REQUIRE_EQ(error.message,
                "shape extent at axis 1 is 0; extents must be positive");
}

TK_TEST("Negative extent reports its value") {
  const auto result = Shape::create({2, -7});
  const auto& error = require_error(result, ErrorCode::extent_not_positive);

  TK_REQUIRE_EQ(error.message,
                "shape extent at axis 1 is -7; extents must be positive");
}

TK_TEST("All extents are validated before multiplication") {
  const auto result = Shape::create(
      {std::numeric_limits<std::int64_t>::max(), 3, 0},
      ShapeLimits{std::numeric_limits<std::uint64_t>::max()});

  require_error(result, ErrorCode::extent_not_positive);
}

TK_TEST("Element limit accepts the exact boundary") {
  const auto result = Shape::create({3, 4}, ShapeLimits{12U});

  TK_REQUIRE_EQ(require_shape(result).numel(), 12U);
}

TK_TEST("Element limit rejects one beyond the boundary") {
  const auto result = Shape::create({3, 4}, ShapeLimits{11U});
  const auto& error =
      require_error(result, ErrorCode::element_limit_exceeded);

  TK_REQUIRE_EQ(error.message, "shape element count 12 exceeds limit 11");
}

TK_TEST("Scalar still consumes one element of the limit") {
  TK_REQUIRE(Shape::create({}, ShapeLimits{1U}).has_value());
  require_error(Shape::create({}, ShapeLimits{0U}),
                ErrorCode::element_limit_exceeded);
}

TK_TEST("Element arithmetic overflow precedes the resource limit") {
  const auto result = Shape::create(
      {std::numeric_limits<std::int64_t>::max(), 3}, ShapeLimits{8U});

  require_error(result, ErrorCode::element_count_overflow);
}

TK_TEST("Shape equality is structural") {
  const Shape left = require_shape(Shape::create({2, 3}));
  const Shape same =
      require_shape(Shape::create({2, 3}, ShapeLimits{1000U}));
  const Shape reversed = require_shape(Shape::create({3, 2}));
  const Shape singleton = require_shape(Shape::create({1}));

  TK_REQUIRE_EQ(left, same);
  TK_REQUIRE(!(left == reversed));
  TK_REQUIRE(!(Shape::scalar() == singleton));
}

TK_TEST("Small shapes satisfy exhaustive boundary properties") {
  constexpr std::array<std::int64_t, 4U> candidates{1, 2, 3, 7};
  std::size_t visited = 0U;

  for (std::size_t rank = 0U; rank <= tensorkiln::kMaxRank; ++rank) {
    std::size_t combinations = 1U;
    for (std::size_t axis = 0U; axis < rank; ++axis) {
      combinations *= candidates.size();
    }

    for (std::size_t ordinal = 0U; ordinal < combinations; ++ordinal) {
      std::array<std::int64_t, tensorkiln::kMaxRank> extents{};
      std::size_t remainder = ordinal;
      std::uint64_t expected = 1U;
      for (std::size_t axis = 0U; axis < rank; ++axis) {
        const std::size_t choice = remainder % candidates.size();
        remainder /= candidates.size();
        extents[axis] = candidates[choice];
        expected *= static_cast<std::uint64_t>(extents[axis]);
      }

      const auto view = std::span<const std::int64_t>(extents.data(), rank);
      const auto exact = Shape::create(view, ShapeLimits{expected});
      const Shape shape = require_shape(exact);
      TK_REQUIRE_EQ(shape.rank(), rank);
      TK_REQUIRE_EQ(shape.numel(), expected);
      TK_REQUIRE(std::equal(shape.extents().begin(), shape.extents().end(),
                            view.begin(), view.end()));

      const auto below = Shape::create(view, ShapeLimits{expected - 1U});
      require_error(below, ErrorCode::element_limit_exceeded);
      ++visited;
    }
  }

  TK_REQUIRE_EQ(visited, 341U);
}

}  // namespace
