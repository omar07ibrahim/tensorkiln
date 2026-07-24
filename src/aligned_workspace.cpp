#include "aligned_workspace.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>

#include "tensorkiln/arena.hpp"

namespace tensorkiln::detail {
namespace {

constexpr std::byte kPrefixGuard{0xA5U};
constexpr std::byte kSuffixGuard{0x5AU};

}  // namespace

AlignedWorkspace::AlignedWorkspace(const std::uint64_t logical_bytes)
    : logical_bytes_(logical_bytes) {
  if (logical_bytes == 0U) {
    return;
  }
  if (logical_bytes >
      static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) -
          2U * kExecutionGuardBytes) {
    throw std::bad_alloc{};
  }

  const auto logical_size = static_cast<std::size_t>(logical_bytes);
  const std::size_t allocation_size =
      logical_size + 2U * kExecutionGuardBytes;
  allocation_ = static_cast<std::byte*>(::operator new(
      allocation_size,
      std::align_val_t{static_cast<std::size_t>(kArenaAlignmentBytes)}));
  data_ = allocation_ + kExecutionGuardBytes;
  assert(reinterpret_cast<std::uintptr_t>(data_) %
             static_cast<std::uintptr_t>(kArenaAlignmentBytes) ==
         0U);

  std::fill_n(allocation_, kExecutionGuardBytes, kPrefixGuard);
  std::fill_n(data_ + logical_size, kExecutionGuardBytes, kSuffixGuard);
}

AlignedWorkspace::AlignedWorkspace(AlignedWorkspace&& other) noexcept
    : allocation_(other.allocation_),
      data_(other.data_),
      logical_bytes_(other.logical_bytes_) {
  other.allocation_ = nullptr;
  other.data_ = nullptr;
  other.logical_bytes_ = 0U;
}

AlignedWorkspace::~AlignedWorkspace() { release(); }

std::span<std::byte> AlignedWorkspace::bytes() noexcept {
  return {data_, static_cast<std::size_t>(logical_bytes_)};
}

std::span<const std::byte> AlignedWorkspace::bytes() const noexcept {
  return {data_, static_cast<std::size_t>(logical_bytes_)};
}

bool AlignedWorkspace::guards_intact() const noexcept {
  if (logical_bytes_ == 0U) {
    return allocation_ == nullptr && data_ == nullptr;
  }
  assert(allocation_ != nullptr);
  assert(data_ != nullptr);
  const auto prefix = std::span<const std::byte>{allocation_,
                                                kExecutionGuardBytes};
  const auto suffix = std::span<const std::byte>{
      data_ + static_cast<std::size_t>(logical_bytes_),
      kExecutionGuardBytes};
  return std::all_of(prefix.begin(), prefix.end(), [](const std::byte value) {
           return value == kPrefixGuard;
         }) &&
         std::all_of(suffix.begin(), suffix.end(), [](const std::byte value) {
           return value == kSuffixGuard;
         });
}

void AlignedWorkspace::corrupt_prefix_for_test() noexcept {
  if (allocation_ != nullptr) {
    allocation_[kExecutionGuardBytes - 1U] = std::byte{0U};
  }
}

void AlignedWorkspace::corrupt_suffix_for_test() noexcept {
  if (data_ != nullptr) {
    data_[static_cast<std::size_t>(logical_bytes_)] = std::byte{0U};
  }
}

void AlignedWorkspace::release() noexcept {
  if (allocation_ != nullptr) {
    ::operator delete(
        allocation_,
        std::align_val_t{static_cast<std::size_t>(kArenaAlignmentBytes)});
  }
  allocation_ = nullptr;
  data_ = nullptr;
  logical_bytes_ = 0U;
}

}  // namespace tensorkiln::detail
