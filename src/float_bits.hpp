#pragma once

#include <cstring>
#include <span>
#include <vector>

namespace tensorkiln::detail {

[[nodiscard]] inline std::vector<float> copy_float_bits(
    const std::span<const float> source) {
  std::vector<float> result(source.size());
  if (!source.empty()) {
    static_cast<void>(
        std::memcpy(result.data(), source.data(), source.size_bytes()));
  }
  return result;
}

}  // namespace tensorkiln::detail
