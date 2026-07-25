#pragma once

#include <span>

#include "tensorkiln/execution_plan.hpp"

namespace tensorkiln::detail {

// Performs any process-local standard-library initialization required by
// kernels before the allocation-free run() boundary begins.
void prepare_dense_kernel_runtime(
    std::span<const ExecutionStep> steps) noexcept;

// All metadata is sealed by ExecutionPlanVerifier. The dispatch writes exactly
// the output tensor payload and performs no validation or heap allocation.
void execute_dense_kernel(
    const ExecutionStep& step, std::span<const PlanValue> values,
    std::span<const float* const> value_data, float* output) noexcept;

}  // namespace tensorkiln::detail
