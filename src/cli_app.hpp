#pragma once

#include <iosfwd>
#include <span>
#include <string_view>

namespace tensorkiln::cli {

inline constexpr int kExitSuccess = 0;
inline constexpr int kExitUsage = 2;
inline constexpr int kExitBuildFailure = 3;
inline constexpr int kExitRunFailure = 4;
inline constexpr int kExitReferenceMismatch = 5;
inline constexpr int kExitInternalFailure = 70;

// Dispatches one command without consulting process-global argv, locale,
// filesystem state, or environment variables. This entry point is private to
// the TensorKiln executable and its contract tests.
[[nodiscard]] int run(std::span<const std::string_view> arguments,
                      std::ostream& output, std::ostream& error);

}  // namespace tensorkiln::cli
