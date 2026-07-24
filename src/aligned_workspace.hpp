#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

namespace tensorkiln::detail {

inline constexpr std::size_t kExecutionGuardBytes = 64U;

class AlignedWorkspace final {
 public:
  explicit AlignedWorkspace(std::uint64_t logical_bytes);

  AlignedWorkspace(const AlignedWorkspace&) = delete;
  AlignedWorkspace(AlignedWorkspace&& other) noexcept;
  AlignedWorkspace& operator=(const AlignedWorkspace&) = delete;
  AlignedWorkspace& operator=(AlignedWorkspace&&) = delete;
  ~AlignedWorkspace();

  [[nodiscard]] std::uint64_t logical_bytes() const noexcept {
    return logical_bytes_;
  }
  [[nodiscard]] std::span<std::byte> bytes() noexcept;
  [[nodiscard]] std::span<const std::byte> bytes() const noexcept;
  [[nodiscard]] bool guards_intact() const noexcept;

  // Internal fault hooks exercise the detector without exposing mutable guard
  // storage through the public execution API.
  void corrupt_prefix_for_test() noexcept;
  void corrupt_suffix_for_test() noexcept;

 private:
  void release() noexcept;

  std::byte* allocation_ = nullptr;
  std::byte* data_ = nullptr;
  std::uint64_t logical_bytes_ = 0U;
};

}  // namespace tensorkiln::detail
