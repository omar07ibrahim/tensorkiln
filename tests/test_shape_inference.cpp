#include "test.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <span>
#include <string>

#include "tensorkiln/shape_inference.hpp"

namespace {

using tensorkiln::ErrorCode;
using tensorkiln::Shape;
using tensorkiln::ShapeLimits;

[[nodiscard]] Shape make_shape(
    const std::initializer_list<std::int64_t> extents) {
  const auto result = Shape::create(extents);
  TK_REQUIRE(result.value_if() != nullptr);
  return *result.value_if();
}

[[nodiscard]] Shape make_unbounded_shape(
    const std::initializer_list<std::int64_t> extents) {
  const auto result = Shape::create(
      extents, ShapeLimits{std::numeric_limits<std::uint64_t>::max()});
  TK_REQUIRE(result.value_if() != nullptr);
  return *result.value_if();
}

[[nodiscard]] Shape require_shape(const tensorkiln::Result<Shape>& result) {
  TK_REQUIRE(result.value_if() != nullptr);
  return *result.value_if();
}

const tensorkiln::Diagnostic& require_error(
    const tensorkiln::Result<Shape>& result, const ErrorCode code) {
  TK_REQUIRE(result.error_if() != nullptr);
  TK_REQUIRE_EQ(result.error_if()->code, code);
  return *result.error_if();
}

TK_TEST("Broadcast preserves equal shapes") {
  const Shape shape = make_shape({2, 3, 4});

  TK_REQUIRE_EQ(require_shape(tensorkiln::infer_broadcast_shape(shape, shape)),
                shape);
}

TK_TEST("Scalar broadcasts across every axis") {
  const Shape matrix = make_shape({2, 3});

  TK_REQUIRE_EQ(require_shape(tensorkiln::infer_broadcast_shape(
                    Shape::scalar(), matrix)),
                matrix);
  TK_REQUIRE_EQ(require_shape(tensorkiln::infer_broadcast_shape(
                    matrix, Shape::scalar())),
                matrix);
}

TK_TEST("Two scalars broadcast to a scalar") {
  TK_REQUIRE_EQ(require_shape(tensorkiln::infer_broadcast_shape(
                    Shape::scalar(), Shape::scalar())),
                Shape::scalar());
  require_error(tensorkiln::infer_broadcast_shape(
                    Shape::scalar(), Shape::scalar(), ShapeLimits{0U}),
                ErrorCode::element_limit_exceeded);
}

TK_TEST("Trailing broadcast expands singleton dimensions") {
  const Shape left = make_shape({2, 1, 4});
  const Shape right = make_shape({3, 4});

  TK_REQUIRE_EQ(
      require_shape(tensorkiln::infer_broadcast_shape(left, right)),
      make_shape({2, 3, 4}));
  TK_REQUIRE_EQ(
      require_shape(tensorkiln::infer_broadcast_shape(right, left)),
      make_shape({2, 3, 4}));
}

TK_TEST("Incompatible broadcast identifies the output axis") {
  const Shape left = make_shape({2, 3});
  const Shape right = make_shape({4, 3});
  const auto result = tensorkiln::infer_broadcast_shape(left, right);
  const auto& error = require_error(result, ErrorCode::broadcast_incompatible);

  TK_REQUIRE_EQ(error.message,
                "elementwise cannot broadcast [2,3] with [4,3]: output axis "
                "0 has extents 2 and 4");
}

TK_TEST("Broadcast reports the rightmost incompatible output axis") {
  const auto result = tensorkiln::infer_broadcast_shape(make_shape({2, 3}),
                                                         make_shape({4, 5}));
  const auto& error = require_error(result, ErrorCode::broadcast_incompatible);

  TK_REQUIRE_EQ(error.message,
                "elementwise cannot broadcast [2,3] with [4,5]: output axis "
                "1 has extents 3 and 5");
}

TK_TEST("Broadcast semantic failure precedes a zero element limit") {
  const auto result = tensorkiln::infer_broadcast_shape(
      make_shape({2}), make_shape({3}), ShapeLimits{0U});

  require_error(result, ErrorCode::broadcast_incompatible);
}

TK_TEST("Broadcast element overflow precedes its element limit") {
  const auto result = tensorkiln::infer_broadcast_shape(
      make_unbounded_shape({std::numeric_limits<std::int64_t>::max(), 1}),
      make_unbounded_shape({1, 3}), ShapeLimits{1U});

  require_error(result, ErrorCode::element_count_overflow);
}

TK_TEST("Broadcast output obeys the element ceiling") {
  TK_REQUIRE(tensorkiln::infer_broadcast_shape(
                 make_shape({8, 1}), make_shape({1, 8}), ShapeLimits{64U})
                 .has_value());
  const auto result = tensorkiln::infer_broadcast_shape(
      make_shape({8, 1}), make_shape({1, 8}), ShapeLimits{63U});

  require_error(result, ErrorCode::element_limit_exceeded);
}

TK_TEST("Rank-two matmul infers matrix dimensions") {
  const auto result = tensorkiln::infer_matmul_shape(make_shape({2, 3}),
                                                      make_shape({3, 5}));

  TK_REQUIRE_EQ(require_shape(result), make_shape({2, 5}));
}

TK_TEST("Batched matmul broadcasts both batch prefixes") {
  const auto result = tensorkiln::infer_matmul_shape(
      make_shape({2, 1, 3, 5}), make_shape({1, 4, 5, 7}));

  TK_REQUIRE_EQ(require_shape(result), make_shape({2, 4, 3, 7}));
}

TK_TEST("Matrix broadcasts across a rank-four batch") {
  const auto result = tensorkiln::infer_matmul_shape(
      make_shape({3, 5}), make_shape({2, 1, 5, 7}));

  TK_REQUIRE_EQ(require_shape(result), make_shape({2, 1, 3, 7}));
}

TK_TEST("Matmul supports every rank pairing from two through four") {
  for (std::size_t left_rank = 2U; left_rank <= 4U; ++left_rank) {
    for (std::size_t right_rank = 2U; right_rank <= 4U; ++right_rank) {
      std::array<std::int64_t, tensorkiln::kMaxRank> left_extents{
          1, 1, 1, 1};
      std::array<std::int64_t, tensorkiln::kMaxRank> right_extents{
          1, 1, 1, 1};
      left_extents[left_rank - 2U] = 2;
      left_extents[left_rank - 1U] = 3;
      right_extents[right_rank - 2U] = 3;
      right_extents[right_rank - 1U] = 5;

      const auto left = Shape::create(std::span<const std::int64_t>(
          left_extents.data(), left_rank));
      const auto right = Shape::create(std::span<const std::int64_t>(
          right_extents.data(), right_rank));
      TK_REQUIRE(left.value_if() != nullptr);
      TK_REQUIRE(right.value_if() != nullptr);

      const Shape output = require_shape(tensorkiln::infer_matmul_shape(
          *left.value_if(), *right.value_if()));
      TK_REQUIRE_EQ(output.rank(), std::max(left_rank, right_rank));
      TK_REQUIRE_EQ(output.extents()[output.rank() - 2U], 2);
      TK_REQUIRE_EQ(output.extents()[output.rank() - 1U], 5);
    }
  }
}

TK_TEST("Matmul rejects a rank-one left operand") {
  const auto result = tensorkiln::infer_matmul_shape(make_shape({3}),
                                                      make_shape({3, 2}));
  const auto& error =
      require_error(result, ErrorCode::matmul_rank_unsupported);

  TK_REQUIRE_EQ(error.message,
                "matmul left rank 1 is unsupported; expected rank 2 through 4");
}

TK_TEST("Matmul rejects a rank-zero right operand") {
  const auto result = tensorkiln::infer_matmul_shape(make_shape({2, 3}),
                                                      Shape::scalar());
  const auto& error =
      require_error(result, ErrorCode::matmul_rank_unsupported);

  TK_REQUIRE_EQ(
      error.message,
      "matmul right rank 0 is unsupported; expected rank 2 through 4");
}

TK_TEST("Matmul checks the left rank before the right rank") {
  const auto result = tensorkiln::infer_matmul_shape(Shape::scalar(),
                                                      make_shape({3}));
  const auto& error =
      require_error(result, ErrorCode::matmul_rank_unsupported);

  TK_REQUIRE_EQ(error.message,
                "matmul left rank 0 is unsupported; expected rank 2 through 4");
}

TK_TEST("Matmul reports its K mismatch before batch mismatch") {
  const auto result = tensorkiln::infer_matmul_shape(
      make_shape({2, 3, 4}), make_shape({5, 6, 7}));
  const auto& error = require_error(
      result, ErrorCode::matmul_inner_dimension_mismatch);

  TK_REQUIRE_EQ(error.message,
                "matmul inner dimensions differ: left K is 4 but right K is 6");
}

TK_TEST("Matmul K mismatch precedes a zero element limit") {
  const auto result = tensorkiln::infer_matmul_shape(
      make_shape({2, 3}), make_shape({4, 5}), ShapeLimits{0U});

  require_error(result, ErrorCode::matmul_inner_dimension_mismatch);
}

TK_TEST("Matmul rejects incompatible batch prefixes") {
  const auto result = tensorkiln::infer_matmul_shape(
      make_shape({2, 3, 4}), make_shape({5, 4, 7}));
  const auto& error = require_error(result, ErrorCode::broadcast_incompatible);

  TK_REQUIRE_EQ(error.message,
                "matmul batch cannot broadcast [2,3,4] with [5,4,7]: output "
                "axis 0 has extents 2 and 5");
}

TK_TEST("Matmul reports the rightmost incompatible batch output axis") {
  const auto result = tensorkiln::infer_matmul_shape(
      make_shape({2, 3, 4, 5}), make_shape({4, 5, 5, 7}));
  const auto& error = require_error(result, ErrorCode::broadcast_incompatible);

  TK_REQUIRE_EQ(error.message,
                "matmul batch cannot broadcast [2,3,4,5] with [4,5,5,7]: "
                "output axis 1 has extents 3 and 5");
}

TK_TEST("Matmul batch mismatch precedes a zero element limit") {
  const auto result = tensorkiln::infer_matmul_shape(
      make_shape({2, 3, 4}), make_shape({5, 4, 7}), ShapeLimits{0U});

  require_error(result, ErrorCode::broadcast_incompatible);
}

TK_TEST("Matmul element overflow precedes its element limit") {
  const auto result = tensorkiln::infer_matmul_shape(
      make_unbounded_shape({std::numeric_limits<std::int64_t>::max(), 1}),
      make_unbounded_shape({1, 3}), ShapeLimits{1U});

  require_error(result, ErrorCode::element_count_overflow);
}

TK_TEST("Matmul output obeys the element ceiling") {
  TK_REQUIRE(tensorkiln::infer_matmul_shape(
                 make_shape({8, 4}), make_shape({4, 8}), ShapeLimits{64U})
                 .has_value());
  const auto result = tensorkiln::infer_matmul_shape(
      make_shape({8, 4}), make_shape({4, 8}), ShapeLimits{63U});

  require_error(result, ErrorCode::element_limit_exceeded);
}

}  // namespace
