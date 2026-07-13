#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

#include "tensorkiln/shape.hpp"

namespace tensorkiln {

enum class ElementType : std::uint8_t {
  f32,
};

inline constexpr std::uint64_t kDefaultMaxTensorBytes = UINT64_C(1) << 28U;

struct TensorLimits final {
  std::uint64_t max_bytes = kDefaultMaxTensorBytes;
};

class TensorType final {
 public:
  [[nodiscard]] static Result<TensorType> create(
      Shape shape, ElementType element_type = ElementType::f32,
      TensorLimits limits = TensorLimits{});

  [[nodiscard]] const Shape& shape() const noexcept { return shape_; }
  [[nodiscard]] ElementType element_type() const noexcept {
    return element_type_;
  }
  [[nodiscard]] std::uint64_t numel() const noexcept {
    return shape_.numel();
  }
  [[nodiscard]] std::size_t byte_count() const noexcept { return byte_count_; }
  [[nodiscard]] std::string to_string() const;

  friend bool operator==(const TensorType&, const TensorType&) noexcept =
      default;

 private:
  TensorType(Shape shape, ElementType element_type,
             std::size_t byte_count) noexcept;

  Shape shape_;
  ElementType element_type_;
  std::size_t byte_count_;
};

[[nodiscard]] std::string_view element_type_name(
    ElementType element_type) noexcept;

}  // namespace tensorkiln
