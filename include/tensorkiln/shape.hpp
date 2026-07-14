#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <span>
#include <string>

#include "tensorkiln/result.hpp"

namespace tensorkiln {

inline constexpr std::size_t kMaxRank = 4U;
inline constexpr std::uint64_t kDefaultMaxElements = UINT64_C(1) << 26U;

struct ShapeLimits final {
  std::uint64_t max_elements = kDefaultMaxElements;

  friend bool operator==(const ShapeLimits&, const ShapeLimits&) noexcept =
      default;
};

class Shape final {
 public:
  [[nodiscard]] static Shape scalar() noexcept;

  [[nodiscard]] static Result<Shape> create(
      std::span<const std::int64_t> extents,
      ShapeLimits limits = ShapeLimits{});

  [[nodiscard]] static Result<Shape> create(
      std::initializer_list<std::int64_t> extents,
      ShapeLimits limits = ShapeLimits{});

  [[nodiscard]] std::size_t rank() const noexcept { return rank_; }
  [[nodiscard]] bool is_scalar() const noexcept { return rank_ == 0U; }

  [[nodiscard]] std::span<const std::int64_t> extents() const noexcept {
    return std::span<const std::int64_t>(extents_.data(), rank_);
  }

  [[nodiscard]] std::optional<std::int64_t> extent(
      std::size_t axis) const noexcept;

  [[nodiscard]] std::uint64_t numel() const noexcept { return numel_; }
  [[nodiscard]] std::string to_string() const;

  friend bool operator==(const Shape&, const Shape&) noexcept = default;

 private:
  Shape(std::array<std::int64_t, kMaxRank> extents, std::size_t rank,
        std::uint64_t numel) noexcept;

  std::array<std::int64_t, kMaxRank> extents_{};
  std::size_t rank_ = 0U;
  std::uint64_t numel_ = 1U;
};

}  // namespace tensorkiln
