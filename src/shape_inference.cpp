#include "tensorkiln/shape_inference.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>

namespace tensorkiln {
namespace {

inline constexpr std::size_t kMinMatMulRank = 2U;
inline constexpr std::size_t kMaxMatMulRank = 4U;

[[nodiscard]] std::int64_t trailing_extent(const Shape& shape,
                                           const std::size_t offset) {
  if (offset >= shape.rank()) {
    return 1;
  }
  return shape.extents()[shape.rank() - 1U - offset];
}

[[nodiscard]] Diagnostic broadcast_error(
    const Shape& left, const Shape& right, const std::size_t aligned_axis,
    const std::int64_t left_extent, const std::int64_t right_extent,
    const std::string_view context) {
  return Diagnostic{
      ErrorCode::broadcast_incompatible,
      std::string(context) + " cannot broadcast " + left.to_string() +
          " with " + right.to_string() + ": output axis " +
          std::to_string(aligned_axis) + " has extents " +
          std::to_string(left_extent) + " and " +
          std::to_string(right_extent),
  };
}

[[nodiscard]] Diagnostic matmul_rank_error(const std::string_view operand,
                                           const std::size_t rank) {
  return Diagnostic{
      ErrorCode::matmul_rank_unsupported,
      "matmul " + std::string(operand) + " rank " + std::to_string(rank) +
          " is unsupported; expected rank 2 through 4",
  };
}

[[nodiscard]] Diagnostic matmul_inner_error(const std::int64_t left_extent,
                                            const std::int64_t right_extent) {
  return Diagnostic{
      ErrorCode::matmul_inner_dimension_mismatch,
      "matmul inner dimensions differ: left K is " +
          std::to_string(left_extent) + " but right K is " +
          std::to_string(right_extent),
  };
}

}  // namespace

Result<Shape> infer_broadcast_shape(const Shape& left, const Shape& right,
                                    const ShapeLimits limits) {
  const std::size_t output_rank = std::max(left.rank(), right.rank());
  std::array<std::int64_t, kMaxRank> output{};

  for (std::size_t offset = 0U; offset < output_rank; ++offset) {
    const std::int64_t left_extent = trailing_extent(left, offset);
    const std::int64_t right_extent = trailing_extent(right, offset);
    const std::size_t output_axis = output_rank - 1U - offset;

    if (left_extent == right_extent || left_extent == 1) {
      output[output_axis] = right_extent;
    } else if (right_extent == 1) {
      output[output_axis] = left_extent;
    } else {
      return Result<Shape>::failure(broadcast_error(
          left, right, output_axis, left_extent, right_extent, "elementwise"));
    }
  }

  return Shape::create(
      std::span<const std::int64_t>(output.data(), output_rank), limits);
}

Result<Shape> infer_matmul_shape(const Shape& left, const Shape& right,
                                 const ShapeLimits limits) {
  if (left.rank() < kMinMatMulRank || left.rank() > kMaxMatMulRank) {
    return Result<Shape>::failure(matmul_rank_error("left", left.rank()));
  }
  if (right.rank() < kMinMatMulRank || right.rank() > kMaxMatMulRank) {
    return Result<Shape>::failure(matmul_rank_error("right", right.rank()));
  }

  const auto left_extents = left.extents();
  const auto right_extents = right.extents();
  const std::int64_t left_k = left_extents[left.rank() - 1U];
  const std::int64_t right_k = right_extents[right.rank() - 2U];
  if (left_k != right_k) {
    return Result<Shape>::failure(matmul_inner_error(left_k, right_k));
  }

  const std::size_t left_batch_rank = left.rank() - 2U;
  const std::size_t right_batch_rank = right.rank() - 2U;
  const std::size_t output_batch_rank =
      std::max(left_batch_rank, right_batch_rank);
  const std::size_t output_rank = output_batch_rank + 2U;
  std::array<std::int64_t, kMaxRank> output{};

  for (std::size_t offset = 0U; offset < output_batch_rank; ++offset) {
    const std::int64_t left_extent =
        offset < left_batch_rank
            ? left_extents[left_batch_rank - 1U - offset]
            : 1;
    const std::int64_t right_extent =
        offset < right_batch_rank
            ? right_extents[right_batch_rank - 1U - offset]
            : 1;
    const std::size_t output_axis = output_batch_rank - 1U - offset;

    if (left_extent == right_extent || left_extent == 1) {
      output[output_axis] = right_extent;
    } else if (right_extent == 1) {
      output[output_axis] = left_extent;
    } else {
      return Result<Shape>::failure(broadcast_error(
          left, right, output_axis, left_extent, right_extent,
          "matmul batch"));
    }
  }

  output[output_rank - 2U] = left_extents[left.rank() - 2U];
  output[output_rank - 1U] = right_extents[right.rank() - 1U];
  return Shape::create(
      std::span<const std::int64_t>(output.data(), output_rank), limits);
}

}  // namespace tensorkiln
