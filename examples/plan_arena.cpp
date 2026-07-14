#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "tensorkiln/arena.hpp"

namespace {

template <typename T>
[[nodiscard]] T unwrap(tensorkiln::Result<T> result) {
  if (result.value_if() == nullptr) {
    const tensorkiln::Diagnostic& error = *result.error_if();
    throw std::runtime_error(
        std::string(tensorkiln::error_code_name(error.code)) + ": " +
        error.message);
  }
  return std::move(*result.value_if());
}

void require_expected_plan(const tensorkiln::ArenaPlan& plan) {
  constexpr std::array<std::uint64_t, 4U> expected_offsets{{
      0U,
      128U,
      128U,
      0U,
  }};
  constexpr std::array<std::uint64_t, 4U> expected_reservations{{
      128U,
      64U,
      64U,
      128U,
  }};

  for (std::size_t index = 0U; index < expected_offsets.size(); ++index) {
    const tensorkiln::ArenaAllocation* allocation =
        plan.allocation_at(static_cast<std::uint32_t>(index));
    if (allocation == nullptr ||
        allocation->offset_bytes() != expected_offsets[index] ||
        allocation->reserved_bytes() != expected_reservations[index]) {
      throw std::runtime_error("arena planner changed its documented layout");
    }
  }

  const tensorkiln::ArenaPlanStats& stats = plan.stats();
  if (stats.buffer_count != 4U || stats.total_payload_bytes != 272U ||
      stats.total_reserved_bytes != 384U ||
      stats.peak_live_reserved_bytes != 192U ||
      stats.workspace_bytes != 192U) {
    throw std::runtime_error("arena planner changed its documented statistics");
  }
}

}  // namespace

int main() {
  try {
    // These requests model storage roots after a future lowering stage has
    // already derived sizes, aliases, and half-open execution lifetimes.
    constexpr std::array<tensorkiln::ArenaBufferRequest, 4U> requests{{
        {96U, 0U, 2U},
        {64U, 0U, 1U},
        {32U, 1U, 3U},
        {80U, 2U, 4U},
    }};

    const tensorkiln::ArenaPlan plan =
        unwrap(tensorkiln::ArenaPlanner::run(requests));
    require_expected_plan(plan);

    std::vector<tensorkiln::ArenaPlacement> reversed_placements;
    reversed_placements.reserve(plan.allocations().size());
    for (auto allocation = plan.allocations().rbegin();
         allocation != plan.allocations().rend(); ++allocation) {
      reversed_placements.push_back(tensorkiln::ArenaPlacement{
          allocation->buffer_ordinal(),
          allocation->offset_bytes(),
      });
    }
    const tensorkiln::ArenaPlan independently_verified = unwrap(
        tensorkiln::ArenaPlacementVerifier::verify(requests,
                                                   reversed_placements));
    if (independently_verified.dump() != plan.dump()) {
      throw std::runtime_error(
          "placement verification did not canonicalize the same plan");
    }

    std::cout << "=== verified interval arena plan ===\n" << plan.dump();
    std::cout << "naive_separate_reservations_bytes="
              << plan.stats().total_reserved_bytes << '\n';
    std::cout << "reused_workspace_bytes=" << plan.workspace_bytes() << '\n';
    std::cout << "verified: two boundary reuses, 192 bytes of workspace "
                 "for 384 bytes of aligned reservations\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "TensorKiln arena example failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
