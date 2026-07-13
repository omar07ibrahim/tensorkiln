#include "test.hpp"

#include <cstdint>
#include <initializer_list>
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

TK_TEST("Incompatible broadcast identifies the aligned axis") {
  const Shape left = make_shape({2, 3});
  const Shape right = make_shape({4, 3});
  const auto result = tensorkiln::infer_broadcast_shape(left, right);
  const auto& error = require_error(result, ErrorCode::broadcast_incompatible);

  TK_REQUIRE_EQ(error.message,
                "elementwise cannot broadcast [2,3] with [4,3]: aligned axis "
                "0 has extents 2 and 4");
}

TK_TEST("Broadcast output obeys the element ceiling") {
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

TK_TEST("Matmul reports its K mismatch before batch mismatch") {
  const auto result = tensorkiln::infer_matmul_shape(
      make_shape({2, 3, 4}), make_shape({5, 6, 7}));
  const auto& error = require_error(
      result, ErrorCode::matmul_inner_dimension_mismatch);

  TK_REQUIRE_EQ(error.message,
                "matmul inner dimensions differ: left K is 4 but right K is 6");
}

TK_TEST("Matmul rejects incompatible batch prefixes") {
  const auto result = tensorkiln::infer_matmul_shape(
      make_shape({2, 3, 4}), make_shape({5, 4, 7}));
  const auto& error = require_error(result, ErrorCode::broadcast_incompatible);

  TK_REQUIRE_EQ(error.message,
                "matmul batch cannot broadcast [2,3,4] with [5,4,7]: aligned "
                "axis 0 has extents 2 and 5");
}

TK_TEST("Matmul output obeys the element ceiling") {
  const auto result = tensorkiln::infer_matmul_shape(
      make_shape({8, 4}), make_shape({4, 8}), ShapeLimits{63U});

  require_error(result, ErrorCode::element_limit_exceeded);
}

}  // namespace
