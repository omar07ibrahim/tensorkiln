#include "tensorkiln/shape.hpp"

#include <limits>
#include <utility>

namespace tensorkiln {
namespace {

[[nodiscard]] Diagnostic rank_error(const std::size_t actual) {
  return Diagnostic{
      ErrorCode::rank_limit_exceeded,
      "shape rank " + std::to_string(actual) + " exceeds limit " +
          std::to_string(kMaxRank),
  };
}

[[nodiscard]] Diagnostic extent_error(const std::size_t axis,
                                      const std::int64_t extent) {
  return Diagnostic{
      ErrorCode::extent_not_positive,
      "shape extent at axis " + std::to_string(axis) + " is " +
          std::to_string(extent) + "; extents must be positive",
  };
}

[[nodiscard]] Diagnostic element_overflow_error() {
  return Diagnostic{
      ErrorCode::element_count_overflow,
      "shape element count exceeds uint64 capacity",
  };
}

[[nodiscard]] Diagnostic element_limit_error(const std::uint64_t actual,
                                             const std::uint64_t limit) {
  return Diagnostic{
      ErrorCode::element_limit_exceeded,
      "shape element count " + std::to_string(actual) + " exceeds limit " +
          std::to_string(limit),
  };
}

}  // namespace

Shape::Shape(std::array<std::int64_t, kMaxRank> extents,
             const std::size_t rank, const std::uint64_t numel) noexcept
    : extents_(std::move(extents)), rank_(rank), numel_(numel) {}

Shape Shape::scalar() noexcept {
  return Shape(std::array<std::int64_t, kMaxRank>{}, 0U, 1U);
}

Result<Shape> Shape::create(const std::span<const std::int64_t> extents,
                            const ShapeLimits limits) {
  if (extents.size() > kMaxRank) {
    return Result<Shape>::failure(rank_error(extents.size()));
  }

  std::array<std::int64_t, kMaxRank> storage{};
  for (std::size_t axis = 0U; axis < extents.size(); ++axis) {
    const std::int64_t extent = extents[axis];
    if (extent <= 0) {
      return Result<Shape>::failure(extent_error(axis, extent));
    }
    storage[axis] = extent;
  }

  std::uint64_t numel = 1U;
  for (std::size_t axis = 0U; axis < extents.size(); ++axis) {
    const auto extent = static_cast<std::uint64_t>(storage[axis]);
    if (numel > std::numeric_limits<std::uint64_t>::max() / extent) {
      return Result<Shape>::failure(element_overflow_error());
    }
    numel *= extent;
  }

  if (numel > limits.max_elements) {
    return Result<Shape>::failure(
        element_limit_error(numel, limits.max_elements));
  }

  return Result<Shape>::success(
      Shape(std::move(storage), extents.size(), numel));
}

Result<Shape> Shape::create(
    const std::initializer_list<std::int64_t> extents,
    const ShapeLimits limits) {
  return create(std::span<const std::int64_t>(extents.begin(), extents.size()),
                limits);
}

std::optional<std::int64_t> Shape::extent(const std::size_t axis) const
    noexcept {
  if (axis >= rank_) {
    return std::nullopt;
  }
  return extents_[axis];
}

std::string Shape::to_string() const {
  std::string result{"["};
  for (std::size_t axis = 0U; axis < rank_; ++axis) {
    if (axis != 0U) {
      result += ',';
    }
    result += std::to_string(extents_[axis]);
  }
  result += ']';
  return result;
}

}  // namespace tensorkiln
