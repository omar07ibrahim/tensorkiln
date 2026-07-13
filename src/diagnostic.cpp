#include "tensorkiln/diagnostic.hpp"

namespace tensorkiln {

std::string_view error_code_name(const ErrorCode code) noexcept {
  switch (code) {
    case ErrorCode::rank_limit_exceeded:
      return "rank_limit_exceeded";
    case ErrorCode::extent_not_positive:
      return "extent_not_positive";
    case ErrorCode::element_count_overflow:
      return "element_count_overflow";
    case ErrorCode::element_limit_exceeded:
      return "element_limit_exceeded";
    case ErrorCode::unsupported_element_type:
      return "unsupported_element_type";
    case ErrorCode::byte_count_overflow:
      return "byte_count_overflow";
    case ErrorCode::byte_limit_exceeded:
      return "byte_limit_exceeded";
  }
  return "unknown_error";
}

}  // namespace tensorkiln
