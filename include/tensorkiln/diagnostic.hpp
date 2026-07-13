#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace tensorkiln {

enum class ErrorCode : std::uint8_t {
  rank_limit_exceeded,
  extent_not_positive,
  element_count_overflow,
  element_limit_exceeded,
  broadcast_incompatible,
  matmul_rank_unsupported,
  matmul_inner_dimension_mismatch,
  unsupported_element_type,
  byte_count_overflow,
  byte_limit_exceeded,
};

struct Diagnostic final {
  ErrorCode code;
  std::string message;

  friend bool operator==(const Diagnostic&, const Diagnostic&) = default;
};

[[nodiscard]] std::string_view error_code_name(ErrorCode code) noexcept;

}  // namespace tensorkiln
