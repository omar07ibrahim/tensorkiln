#include "test.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "tensorkiln/arena.hpp"

namespace {

using tensorkiln::ArenaAllocation;
using tensorkiln::ArenaBufferRequest;
using tensorkiln::ArenaLimits;
using tensorkiln::ArenaPlacement;
using tensorkiln::ArenaPlacementVerifier;
using tensorkiln::ArenaPlan;
using tensorkiln::ArenaPlanner;
using tensorkiln::ErrorCode;

class DeterministicGenerator final {
 public:
  explicit DeterministicGenerator(const std::uint32_t seed) noexcept
      : state_(static_cast<std::uint64_t>(seed)) {}

  [[nodiscard]] std::uint64_t next() noexcept {
    state_ += UINT64_C(0x9e3779b97f4a7c15);
    std::uint64_t value = state_;
    value = (value ^ (value >> 30U)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27U)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31U);
  }

  [[nodiscard]] std::size_t index(const std::size_t bound) {
    TK_REQUIRE(bound > 0U);
    const std::uint64_t range = static_cast<std::uint64_t>(bound);
    const std::uint64_t threshold = (UINT64_C(0) - range) % range;
    std::uint64_t value = next();
    while (value < threshold) {
      value = next();
    }
    return static_cast<std::size_t>(value % range);
  }

 private:
  std::uint64_t state_;
};

[[nodiscard]] ArenaPlan require_plan(tensorkiln::Result<ArenaPlan> result) {
  TK_REQUIRE(result.value_if() != nullptr);
  return std::move(*result.value_if());
}

// Direct arithmetic is safe for these generated fixtures: payloads are at most
// 2048 bytes, offsets stay below 512 KiB, and cohorts contain at most 4096
// buffers. Production overflow behavior is covered by focused verifier tests.
[[nodiscard]] std::uint64_t reserved_bytes(
    const ArenaBufferRequest& request) {
  return (request.size_bytes + (tensorkiln::kArenaAlignmentBytes - 1U)) &
         ~(tensorkiln::kArenaAlignmentBytes - 1U);
}

[[nodiscard]] bool lifetimes_overlap(const ArenaBufferRequest& left,
                                     const ArenaBufferRequest& right) {
  return left.live_begin_step < right.live_end_step_exclusive &&
         right.live_begin_step < left.live_end_step_exclusive;
}

[[nodiscard]] bool byte_ranges_overlap(const std::uint64_t left_offset,
                                       const std::uint64_t left_size,
                                       const std::uint64_t right_offset,
                                       const std::uint64_t right_size) {
  return left_offset < right_offset + right_size &&
         right_offset < left_offset + left_size;
}

void require_plan_matches_pairwise_oracle(
    const std::vector<ArenaBufferRequest>& requests,
    const ArenaPlan& plan) {
  TK_REQUIRE_EQ(plan.allocations().size(), requests.size());
  std::uint64_t total_payload_bytes = 0U;
  std::uint64_t total_reserved_bytes = 0U;
  std::uint64_t expected_workspace_bytes = 0U;

  for (std::size_t index = 0U; index < requests.size(); ++index) {
    const ArenaBufferRequest& request = requests[index];
    const ArenaAllocation* allocation =
        plan.allocation_at(static_cast<std::uint32_t>(index));
    TK_REQUIRE(allocation != nullptr);
    const std::uint64_t expected_reserved = reserved_bytes(request);
    TK_REQUIRE_EQ(allocation->buffer_ordinal(),
                  static_cast<std::uint32_t>(index));
    TK_REQUIRE_EQ(allocation->payload_bytes(), request.size_bytes);
    TK_REQUIRE_EQ(allocation->reserved_bytes(), expected_reserved);
    TK_REQUIRE_EQ(allocation->live_begin_step(), request.live_begin_step);
    TK_REQUIRE_EQ(allocation->live_end_step_exclusive(),
                  request.live_end_step_exclusive);
    TK_REQUIRE_EQ(allocation->offset_bytes() %
                      tensorkiln::kArenaAlignmentBytes,
                  0U);
    total_payload_bytes += request.size_bytes;
    total_reserved_bytes += expected_reserved;
    expected_workspace_bytes =
        std::max(expected_workspace_bytes,
                 allocation->offset_bytes() + expected_reserved);
  }

  for (std::size_t left = 0U; left < requests.size(); ++left) {
    for (std::size_t right = left + 1U; right < requests.size(); ++right) {
      if (!lifetimes_overlap(requests[left], requests[right])) {
        continue;
      }
      const ArenaAllocation* left_allocation =
          plan.allocation_at(static_cast<std::uint32_t>(left));
      const ArenaAllocation* right_allocation =
          plan.allocation_at(static_cast<std::uint32_t>(right));
      TK_REQUIRE(left_allocation != nullptr);
      TK_REQUIRE(right_allocation != nullptr);
      TK_REQUIRE(!byte_ranges_overlap(
          left_allocation->offset_bytes(), left_allocation->reserved_bytes(),
          right_allocation->offset_bytes(),
          right_allocation->reserved_bytes()));
    }
  }

  std::uint64_t peak_live_reserved_bytes = 0U;
  for (const ArenaBufferRequest& boundary_request : requests) {
    const std::uint32_t step = boundary_request.live_begin_step;
    std::uint64_t live_reserved_bytes = 0U;
    for (const ArenaBufferRequest& request : requests) {
      if (request.live_begin_step <= step &&
          step < request.live_end_step_exclusive) {
        live_reserved_bytes += reserved_bytes(request);
      }
    }
    peak_live_reserved_bytes =
        std::max(peak_live_reserved_bytes, live_reserved_bytes);
  }

  TK_REQUIRE_EQ(plan.stats().buffer_count,
                static_cast<std::uint32_t>(requests.size()));
  TK_REQUIRE_EQ(plan.stats().total_payload_bytes, total_payload_bytes);
  TK_REQUIRE_EQ(plan.stats().total_reserved_bytes, total_reserved_bytes);
  TK_REQUIRE_EQ(plan.stats().peak_live_reserved_bytes,
                peak_live_reserved_bytes);
  TK_REQUIRE_EQ(plan.workspace_bytes(), expected_workspace_bytes);
  TK_REQUIRE(plan.stats().peak_live_reserved_bytes <= plan.workspace_bytes());
}

[[nodiscard]] std::vector<ArenaBufferRequest> make_seeded_requests(
    const std::uint32_t seed, const std::size_t buffer_count,
    const std::uint32_t step_count) {
  DeterministicGenerator generator(seed);
  std::vector<ArenaBufferRequest> requests;
  requests.reserve(buffer_count);
  for (std::size_t index = 0U; index < buffer_count; ++index) {
    const auto begin =
        static_cast<std::uint32_t>(generator.index(step_count));
    const std::size_t remaining =
        static_cast<std::size_t>(step_count - begin);
    const auto duration =
        static_cast<std::uint32_t>(1U + generator.index(remaining));
    std::uint64_t size_bytes = 1U + generator.index(2048U);
    if ((index % 7U) == 0U) {
      size_bytes = 64U;
    } else if ((index % 11U) == 0U) {
      size_bytes = 65U;
    }
    requests.push_back(ArenaBufferRequest{
        size_bytes,
        begin,
        static_cast<std::uint32_t>(begin + duration),
    });
  }
  return requests;
}

TK_TEST("Arena planning is safe and deterministic across seeded schedules") {
  std::set<std::string> placement_signatures;
  std::size_t plans_with_reuse = 0U;
  constexpr std::size_t buffer_count = 96U;
  constexpr std::uint32_t step_count = 64U;

  for (std::uint32_t seed = 1U; seed <= 64U; ++seed) {
    const std::vector<ArenaBufferRequest> requests =
        make_seeded_requests(seed, buffer_count, step_count);
    const ArenaLimits limits{
        static_cast<std::uint32_t>(buffer_count),
        tensorkiln::kDefaultMaxArenaWorkspaceBytes,
    };
    const ArenaPlan first =
        require_plan(ArenaPlanner::run(requests, limits));
    const ArenaPlan repeated =
        require_plan(ArenaPlanner::run(requests, limits));

    require_plan_matches_pairwise_oracle(requests, first);
    TK_REQUIRE(first.workspace_bytes() <= first.stats().total_reserved_bytes);
    TK_REQUIRE_EQ(repeated.stats(), first.stats());
    TK_REQUIRE_EQ(repeated.dump(), first.dump());
    std::string signature = std::to_string(first.workspace_bytes()) + ":";
    for (const ArenaAllocation& allocation : first.allocations()) {
      signature += std::to_string(allocation.offset_bytes()) + ",";
    }
    placement_signatures.insert(std::move(signature));
    if (first.workspace_bytes() < first.stats().total_reserved_bytes) {
      ++plans_with_reuse;
    }

    std::vector<ArenaPlacement> reversed_placements;
    reversed_placements.reserve(first.allocations().size());
    for (auto allocation = first.allocations().rbegin();
         allocation != first.allocations().rend(); ++allocation) {
      reversed_placements.push_back(ArenaPlacement{
          allocation->buffer_ordinal(),
          allocation->offset_bytes(),
      });
    }
    const ArenaPlan reverified = require_plan(
        ArenaPlacementVerifier::verify(requests, reversed_placements, limits));
    TK_REQUIRE_EQ(reverified.dump(), first.dump());
  }

  TK_REQUIRE(placement_signatures.size() >= 60U);
  TK_REQUIRE_EQ(plans_with_reuse, 64U);
}

TK_TEST("Arena verifier agrees with a seeded pairwise placement oracle") {
  constexpr std::size_t buffer_count = 24U;
  constexpr std::uint32_t step_count = 16U;
  std::size_t accepted = 0U;
  std::size_t rejected = 0U;

  for (std::uint32_t seed = 101U; seed <= 164U; ++seed) {
    DeterministicGenerator generator(seed ^ UINT32_C(0xa5a5a5a5));
    std::vector<ArenaBufferRequest> requests =
        make_seeded_requests(seed, buffer_count, step_count);
    std::vector<std::uint64_t> offsets(buffer_count);
    const std::size_t slot_count = 512U + generator.index(7680U);
    for (std::size_t index = 0U; index < buffer_count; ++index) {
      offsets[index] = generator.index(slot_count) *
                       tensorkiln::kArenaAlignmentBytes;
    }

    bool oracle_conflict = false;
    for (std::size_t left = 0U; left < buffer_count; ++left) {
      for (std::size_t right = left + 1U; right < buffer_count; ++right) {
        if (lifetimes_overlap(requests[left], requests[right]) &&
            byte_ranges_overlap(offsets[left], reserved_bytes(requests[left]),
                                offsets[right],
                                reserved_bytes(requests[right]))) {
          oracle_conflict = true;
        }
      }
    }

    std::vector<ArenaPlacement> placements;
    placements.reserve(buffer_count);
    for (std::size_t index = 0U; index < buffer_count; ++index) {
      placements.push_back(ArenaPlacement{
          static_cast<std::uint32_t>(index),
          offsets[index],
      });
    }
    for (std::size_t remaining = placements.size(); remaining > 1U;
         --remaining) {
      const std::size_t selected = generator.index(remaining);
      std::swap(placements[remaining - 1U], placements[selected]);
    }

    auto verified = ArenaPlacementVerifier::verify(
        requests, placements,
        ArenaLimits{static_cast<std::uint32_t>(buffer_count),
                    tensorkiln::kDefaultMaxArenaWorkspaceBytes});
    TK_REQUIRE_EQ(verified.has_value(), !oracle_conflict);
    if (oracle_conflict) {
      TK_REQUIRE(verified.error_if() != nullptr);
      TK_REQUIRE_EQ(verified.error_if()->code,
                    ErrorCode::arena_live_overlap);
      ++rejected;
    } else {
      const ArenaPlan plan = require_plan(std::move(verified));
      require_plan_matches_pairwise_oracle(requests, plan);
      for (std::size_t index = 0U; index < buffer_count; ++index) {
        const ArenaAllocation* allocation =
            plan.allocation_at(static_cast<std::uint32_t>(index));
        TK_REQUIRE(allocation != nullptr);
        TK_REQUIRE_EQ(allocation->offset_bytes(), offsets[index]);
      }
      ++accepted;
    }
  }

  TK_REQUIRE(accepted >= 8U);
  TK_REQUIRE(rejected >= 8U);
  TK_REQUIRE_EQ(accepted + rejected, 64U);
}

TK_TEST("Arena planning scales to the exact default buffer boundary") {
  constexpr std::size_t buffer_count =
      tensorkiln::kDefaultMaxArenaBuffers;
  std::vector<ArenaBufferRequest> requests;
  requests.reserve(buffer_count);
  for (std::size_t index = 0U; index < buffer_count; ++index) {
    const std::uint64_t size_bytes = 1U + (index % 4U) * 64U;
    requests.push_back(ArenaBufferRequest{
        size_bytes,
        static_cast<std::uint32_t>(index),
        static_cast<std::uint32_t>(index + 2U),
    });
  }

  const ArenaPlan plan = require_plan(ArenaPlanner::run(
      requests, ArenaLimits{tensorkiln::kDefaultMaxArenaBuffers,
                            UINT64_C(1) << 20U}));
  TK_REQUIRE_EQ(plan.allocations().size(), buffer_count);
  TK_REQUIRE_EQ(plan.stats().buffer_count,
                tensorkiln::kDefaultMaxArenaBuffers);
  TK_REQUIRE(plan.workspace_bytes() < plan.stats().total_reserved_bytes);
  TK_REQUIRE_EQ(plan.stats().peak_live_reserved_bytes, 448U);
  TK_REQUIRE_EQ(plan.workspace_bytes(), 640U);
  for (std::size_t index = 0U; index < buffer_count; index += 257U) {
    const ArenaAllocation* allocation =
        plan.allocation_at(static_cast<std::uint32_t>(index));
    TK_REQUIRE(allocation != nullptr);
    TK_REQUIRE_EQ(allocation->offset_bytes() %
                      tensorkiln::kArenaAlignmentBytes,
                  0U);
  }
  TK_REQUIRE(plan.allocation_at(
                 tensorkiln::kDefaultMaxArenaBuffers - 1U) != nullptr);

  requests.push_back(ArenaBufferRequest{64U, 0U, 1U});
  const auto one_too_many = ArenaPlanner::run(
      requests, ArenaLimits{tensorkiln::kDefaultMaxArenaBuffers,
                            UINT64_C(1) << 20U});
  TK_REQUIRE(one_too_many.error_if() != nullptr);
  TK_REQUIRE_EQ(one_too_many.error_if()->code,
                ErrorCode::arena_buffer_limit_exceeded);
  TK_REQUIRE_EQ(one_too_many.error_if()->message,
                "arena has 4097 buffer requests; limit is 4096");
}

TK_TEST("Arena planning accepts the maximum half-open step endpoint") {
  const std::vector<ArenaBufferRequest> requests{
      {64U, std::numeric_limits<std::uint32_t>::max() - 2U,
       std::numeric_limits<std::uint32_t>::max() - 1U},
      {64U, std::numeric_limits<std::uint32_t>::max() - 1U,
       std::numeric_limits<std::uint32_t>::max()},
  };
  const ArenaPlan plan = require_plan(ArenaPlanner::run(requests));
  TK_REQUIRE_EQ(plan.allocation_at(0U)->live_begin_step(),
                std::numeric_limits<std::uint32_t>::max() - 2U);
  TK_REQUIRE_EQ(plan.allocation_at(0U)->live_end_step_exclusive(),
                std::numeric_limits<std::uint32_t>::max() - 1U);
  TK_REQUIRE_EQ(plan.allocation_at(1U)->live_begin_step(),
                std::numeric_limits<std::uint32_t>::max() - 1U);
  TK_REQUIRE_EQ(plan.allocation_at(1U)->live_end_step_exclusive(),
                std::numeric_limits<std::uint32_t>::max());
  TK_REQUIRE_EQ(plan.allocation_at(0U)->offset_bytes(), 0U);
  TK_REQUIRE_EQ(plan.allocation_at(1U)->offset_bytes(), 0U);
  TK_REQUIRE_EQ(plan.stats().peak_live_reserved_bytes, 64U);
  TK_REQUIRE_EQ(plan.workspace_bytes(), 64U);
  require_plan_matches_pairwise_oracle(requests, plan);
}

}  // namespace
