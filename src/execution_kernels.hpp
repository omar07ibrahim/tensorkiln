#pragma once

#include <span>

#include "tensorkiln/execution_plan.hpp"

namespace tensorkiln::detail {

// All metadata is sealed by ExecutionPlanVerifier. The dispatch writes exactly
// the output tensor payload and performs no validation or heap allocation.
void execute_dense_kernel(
    const ExecutionStep& step, std::span<const PlanValue> values,
    std::span<const float* const> value_data, float* output) noexcept;

}  // namespace tensorkiln::detail
