#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "tensorkiln/execution_plan.hpp"

namespace tensorkiln::detail {

struct PreparedDenseLayout final {
  std::array<std::uint64_t, kMaxRank> strides_elements{};
  std::size_t rank;
  std::uint64_t elements;
};

struct ExecutionPlanSourceAnalysis final {
  std::uint32_t value_count;
  std::uint32_t input_count;
  std::uint32_t constant_count;
  std::uint32_t step_count;
  std::uint32_t output_count;
  std::uint64_t owned_constant_bytes;
  std::uint64_t scalar_steps;
};

[[nodiscard]] Result<ExecutionPlanSourceAnalysis>
analyze_execution_plan_source(const VerifiedGraph& source,
                              const ExecutionPlanLimits& limits);

[[nodiscard]] Result<PreparedDenseLayout> prepare_dense_layout(
    const TensorType& type, NodeId source_node);

[[nodiscard]] Diagnostic execution_plan_error(ErrorCode code,
                                              std::string message);

}  // namespace tensorkiln::detail
