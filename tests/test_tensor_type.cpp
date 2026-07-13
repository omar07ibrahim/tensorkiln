#include "test.hpp"

#include <cstdint>
#include <limits>
#include <string>

#include "tensorkiln/tensor_type.hpp"

namespace {

using tensorkiln::ElementType;
using tensorkiln::ErrorCode;
using tensorkiln::Shape;
using tensorkiln::ShapeLimits;
using tensorkiln::TensorLimits;
using tensorkiln::TensorType;

[[nodiscard]] Shape require_shape(const tensorkiln::Result<Shape>& result) {
  TK_REQUIRE(result.value_if() != nullptr);
  return *result.value_if();
}

[[nodiscard]] const TensorType& require_type(
    const tensorkiln::Result<TensorType>& result) {
  TK_REQUIRE(result.value_if() != nullptr);
  return *result.value_if();
}

const tensorkiln::Diagnostic& require_error(
    const tensorkiln::Result<TensorType>& result, const ErrorCode code) {
  TK_REQUIRE(result.error_if() != nullptr);
  TK_REQUIRE_EQ(result.error_if()->code, code);
  return *result.error_if();
}

TK_TEST("Scalar f32 type occupies four bytes") {
  const auto result = TensorType::create(Shape::scalar());
  const TensorType& type = require_type(result);

  TK_REQUIRE_EQ(type.element_type(), ElementType::f32);
  TK_REQUIRE_EQ(type.numel(), 1U);
  TK_REQUIRE_EQ(type.byte_count(), sizeof(float));
  TK_REQUIRE_EQ(type.to_string(), "f32[]");
}

TK_TEST("Tensor byte count follows the shape") {
  const auto result =
      TensorType::create(require_shape(Shape::create({2, 3, 4})));
  const TensorType& type = require_type(result);

  TK_REQUIRE_EQ(type.numel(), 24U);
  TK_REQUIRE_EQ(type.byte_count(), 24U * sizeof(float));
  TK_REQUIRE_EQ(type.to_string(), "f32[2,3,4]");
}

TK_TEST("Byte limit accepts the exact boundary") {
  const auto result = TensorType::create(
      require_shape(Shape::create({6})), ElementType::f32,
      TensorLimits{6U * sizeof(float)});

  TK_REQUIRE_EQ(require_type(result).byte_count(), 24U);
}

TK_TEST("Byte limit rejects one byte below the boundary") {
  const auto result = TensorType::create(
      require_shape(Shape::create({6})), ElementType::f32,
      TensorLimits{6U * sizeof(float) - 1U});
  const auto& error = require_error(result, ErrorCode::byte_limit_exceeded);

  TK_REQUIRE_EQ(error.message, "tensor byte count 24 exceeds limit 23");
}

TK_TEST("Byte overflow precedes the byte resource limit") {
  const Shape huge = require_shape(Shape::create(
      {std::numeric_limits<std::int64_t>::max()},
      ShapeLimits{std::numeric_limits<std::uint64_t>::max()}));
  const auto result =
      TensorType::create(huge, ElementType::f32, TensorLimits{1U});

  require_error(result, ErrorCode::byte_count_overflow);
}

TK_TEST("Invalid element enum is rejected") {
  const auto invalid = static_cast<ElementType>(255U);
  const auto result = TensorType::create(Shape::scalar(), invalid);
  const auto& error =
      require_error(result, ErrorCode::unsupported_element_type);

  TK_REQUIRE_EQ(error.message, "unsupported element type value 255");
  TK_REQUIRE_EQ(tensorkiln::element_type_name(invalid), "invalid");
}

TK_TEST("Tensor type equality is structural") {
  const auto left =
      TensorType::create(require_shape(Shape::create({2, 3})));
  const auto same = TensorType::create(
      require_shape(Shape::create({2, 3}, ShapeLimits{100U})),
      ElementType::f32, TensorLimits{100U});
  const auto other =
      TensorType::create(require_shape(Shape::create({3, 2})));

  TK_REQUIRE_EQ(require_type(left), require_type(same));
  TK_REQUIRE(!(require_type(left) == require_type(other)));
}

TK_TEST("Small tensor types satisfy byte-limit boundaries") {
  for (std::int64_t extent = 1; extent <= 64; ++extent) {
    const Shape shape = require_shape(Shape::create({extent}));
    const std::uint64_t bytes =
        static_cast<std::uint64_t>(extent) * sizeof(float);
    const auto exact = TensorType::create(
        shape, ElementType::f32, TensorLimits{bytes});
    TK_REQUIRE_EQ(require_type(exact).byte_count(),
                  static_cast<std::size_t>(bytes));

    const auto below = TensorType::create(
        shape, ElementType::f32, TensorLimits{bytes - 1U});
    require_error(below, ErrorCode::byte_limit_exceeded);
  }
}

}  // namespace
