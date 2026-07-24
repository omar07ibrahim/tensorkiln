#pragma once

#include <cstdint>

#include "tensorkiln/execution.hpp"

namespace tensorkiln::detail {

enum class ExecutionFaultKind : std::uint8_t {
  none,
  write_outside_output,
};

// Internal fault injection used only to prove the optional write-set auditor.
void set_execution_fault_for_test(ExecutionSession& session,
                                  ExecutionFaultKind fault) noexcept;

}  // namespace tensorkiln::detail
