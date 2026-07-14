#include "test.hpp"

#include <memory>
#include <string>

#include "tensorkiln/result.hpp"

namespace {

using tensorkiln::Diagnostic;
using tensorkiln::ErrorCode;
using tensorkiln::Result;

TK_TEST("Result stores a successful value") {
  const Result<int> result = Result<int>::success(37);

  TK_REQUIRE(result.has_value());
  TK_REQUIRE(result.error_if() == nullptr);
  TK_REQUIRE(result.value_if() != nullptr);
  TK_REQUIRE_EQ(*result.value_if(), 37);
}

TK_TEST("Result preserves a typed diagnostic") {
  const Diagnostic expected{
      ErrorCode::element_limit_exceeded,
      "deliberately exact diagnostic",
  };
  const Result<int> result = Result<int>::failure(expected);

  TK_REQUIRE(!result.has_value());
  TK_REQUIRE(result.value_if() == nullptr);
  TK_REQUIRE(result.error_if() != nullptr);
  TK_REQUIRE_EQ(*result.error_if(), expected);
}

TK_TEST("Result accepts a move-only payload") {
  Result<std::unique_ptr<int>> result =
      Result<std::unique_ptr<int>>::success(std::make_unique<int>(91));
  Result<std::unique_ptr<int>> moved = std::move(result);

  TK_REQUIRE(moved.value_if() != nullptr);
  TK_REQUIRE(*moved.value_if() != nullptr);
  TK_REQUIRE_EQ(**moved.value_if(), 91);
}

TK_TEST("Error codes have stable names") {
  TK_REQUIRE_EQ(tensorkiln::error_code_name(ErrorCode::rank_limit_exceeded),
                "rank_limit_exceeded");
  TK_REQUIRE_EQ(
      tensorkiln::error_code_name(ErrorCode::broadcast_incompatible),
      "broadcast_incompatible");
  TK_REQUIRE_EQ(tensorkiln::error_code_name(ErrorCode::byte_limit_exceeded),
                "byte_limit_exceeded");
  TK_REQUIRE_EQ(
      tensorkiln::error_code_name(ErrorCode::input_binding_count_exceeded),
      "input_binding_count_exceeded");
  TK_REQUIRE_EQ(
      tensorkiln::error_code_name(ErrorCode::unsupported_subnormal_mode),
      "unsupported_subnormal_mode");
  TK_REQUIRE_EQ(tensorkiln::error_code_name(
                    ErrorCode::reference_scalar_step_limit_exceeded),
                "reference_scalar_step_limit_exceeded");
}

}  // namespace
