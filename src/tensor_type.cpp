#include "tensorkiln/tensor_type.hpp"

#include <limits>
#include <utility>

namespace tensorkiln {
namespace {

inline constexpr std::uint64_t kF32Bytes = 4U;

static_assert(sizeof(float) == kF32Bytes);
static_assert(std::numeric_limits<float>::is_iec559);

[[nodiscard]] Diagnostic unsupported_type_error(
    const ElementType element_type) {
  return Diagnostic{
      ErrorCode::unsupported_element_type,
      "unsupported element type value " +
          std::to_string(static_cast<unsigned int>(element_type)),
  };
}

[[nodiscard]] Diagnostic byte_overflow_error() {
  return Diagnostic{
      ErrorCode::byte_count_overflow,
      "tensor byte count exceeds addressable size",
  };
}

[[nodiscard]] Diagnostic byte_limit_error(const std::uint64_t actual,
                                          const std::uint64_t limit) {
  return Diagnostic{
      ErrorCode::byte_limit_exceeded,
      "tensor byte count " + std::to_string(actual) + " exceeds limit " +
          std::to_string(limit),
  };
}

}  // namespace

TensorType::TensorType(Shape shape, const ElementType element_type,
                       const std::size_t byte_count) noexcept
    : shape_(std::move(shape)),
      element_type_(element_type),
      byte_count_(byte_count) {}

Result<TensorType> TensorType::create(Shape shape,
                                      const ElementType element_type,
                                      const TensorLimits limits) {
  std::uint64_t width = 0U;
  switch (element_type) {
    case ElementType::f32:
      width = kF32Bytes;
      break;
    default:
      return Result<TensorType>::failure(
          unsupported_type_error(element_type));
  }

  if (shape.numel() > std::numeric_limits<std::uint64_t>::max() / width) {
    return Result<TensorType>::failure(byte_overflow_error());
  }
  const std::uint64_t bytes = shape.numel() * width;

  if (bytes > static_cast<std::uint64_t>(
                  std::numeric_limits<std::size_t>::max())) {
    return Result<TensorType>::failure(byte_overflow_error());
  }
  if (bytes > limits.max_bytes) {
    return Result<TensorType>::failure(
        byte_limit_error(bytes, limits.max_bytes));
  }

  return Result<TensorType>::success(TensorType(
      std::move(shape), element_type, static_cast<std::size_t>(bytes)));
}

std::string TensorType::to_string() const {
  return std::string(element_type_name(element_type_)) + shape_.to_string();
}

std::string_view element_type_name(const ElementType element_type) noexcept {
  switch (element_type) {
    case ElementType::f32:
      return "f32";
  }
  return "invalid";
}

}  // namespace tensorkiln
