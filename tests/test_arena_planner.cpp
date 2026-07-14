#include "test.hpp"

#include <cstdint>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "tensorkiln/arena.hpp"

namespace {

using tensorkiln::ArenaBufferRequest;
using tensorkiln::ArenaLimits;
using tensorkiln::ArenaPlan;
using tensorkiln::ArenaPlanner;
using tensorkiln::ErrorCode;

[[nodiscard]] ArenaPlan require_plan(tensorkiln::Result<ArenaPlan> result) {
  TK_REQUIRE(result.value_if() != nullptr);
  return std::move(*result.value_if());
}

const tensorkiln::Diagnostic& require_error(
    const tensorkiln::Result<ArenaPlan>& result, const ErrorCode code) {
  TK_REQUIRE(result.error_if() != nullptr);
  TK_REQUIRE_EQ(result.error_if()->code, code);
  return *result.error_if();
}

TK_TEST("Arena planner reuses a deterministic two-slot chain") {
  const std::vector<ArenaBufferRequest> requests{
      {64U, 0U, 2U},
      {64U, 1U, 3U},
      {64U, 2U, 4U},
      {64U, 3U, 5U},
  };
  const ArenaPlan plan = require_plan(ArenaPlanner::run(requests));

  TK_REQUIRE_EQ(plan.allocation_at(0U)->offset_bytes(), 0U);
  TK_REQUIRE_EQ(plan.allocation_at(1U)->offset_bytes(), 64U);
  TK_REQUIRE_EQ(plan.allocation_at(2U)->offset_bytes(), 0U);
  TK_REQUIRE_EQ(plan.allocation_at(3U)->offset_bytes(), 64U);
  TK_REQUIRE_EQ(plan.stats().total_reserved_bytes, 256U);
  TK_REQUIRE_EQ(plan.stats().peak_live_reserved_bytes, 128U);
  TK_REQUIRE_EQ(plan.workspace_bytes(), 128U);
}

TK_TEST("Arena planner repeats non-chronological request schedules exactly") {
  const std::vector<ArenaBufferRequest> requests{
      {64U, 3U, 5U},
      {24U, 0U, 3U},
      {32U, 2U, 4U},
      {40U, 1U, 2U},
  };
  const ArenaPlan first = require_plan(ArenaPlanner::run(requests));
  const ArenaPlan second = require_plan(ArenaPlanner::run(requests));

  TK_REQUIRE_EQ(first.dump(), second.dump());
  TK_REQUIRE_EQ(first.allocation_at(0U)->offset_bytes(), 0U);
  TK_REQUIRE_EQ(first.allocation_at(1U)->offset_bytes(), 0U);
  TK_REQUIRE_EQ(first.allocation_at(2U)->offset_bytes(), 64U);
  TK_REQUIRE_EQ(first.allocation_at(3U)->offset_bytes(), 64U);
}

TK_TEST("Arena planner chooses best fit before the lower offset") {
  const std::vector<ArenaBufferRequest> requests{
      {65U, 0U, 2U},
      {64U, 0U, 4U},
      {64U, 0U, 2U},
      {64U, 2U, 3U},
  };
  const ArenaPlan plan = require_plan(ArenaPlanner::run(requests));

  TK_REQUIRE_EQ(plan.allocation_at(0U)->offset_bytes(), 0U);
  TK_REQUIRE_EQ(plan.allocation_at(1U)->offset_bytes(), 128U);
  TK_REQUIRE_EQ(plan.allocation_at(2U)->offset_bytes(), 192U);
  TK_REQUIRE_EQ(plan.allocation_at(3U)->offset_bytes(), 192U);
}

TK_TEST("Arena planner breaks equal-size ties by lower offset") {
  const std::vector<ArenaBufferRequest> requests{
      {64U, 0U, 2U},
      {64U, 0U, 4U},
      {64U, 0U, 2U},
      {64U, 2U, 3U},
  };
  const ArenaPlan plan = require_plan(ArenaPlanner::run(requests));
  TK_REQUIRE_EQ(plan.allocation_at(3U)->offset_bytes(), 0U);
}

TK_TEST("Arena planner splits a larger free block on reuse") {
  const std::vector<ArenaBufferRequest> requests{
      {129U, 0U, 2U},
      {64U, 0U, 5U},
      {64U, 2U, 4U},
      {65U, 3U, 4U},
  };
  const ArenaPlan plan = require_plan(ArenaPlanner::run(requests));

  TK_REQUIRE_EQ(plan.allocation_at(0U)->offset_bytes(), 0U);
  TK_REQUIRE_EQ(plan.allocation_at(1U)->offset_bytes(), 192U);
  TK_REQUIRE_EQ(plan.allocation_at(2U)->offset_bytes(), 0U);
  TK_REQUIRE_EQ(plan.allocation_at(3U)->offset_bytes(), 64U);
  TK_REQUIRE_EQ(plan.workspace_bytes(), 256U);
}

TK_TEST("Arena planner coalesces adjacent free blocks") {
  const std::vector<ArenaBufferRequest> requests{
      {64U, 0U, 2U},
      {64U, 0U, 3U},
      {64U, 0U, 5U},
      {65U, 3U, 4U},
  };
  const ArenaPlan plan = require_plan(ArenaPlanner::run(requests));

  TK_REQUIRE_EQ(plan.allocation_at(0U)->offset_bytes(), 0U);
  TK_REQUIRE_EQ(plan.allocation_at(1U)->offset_bytes(), 64U);
  TK_REQUIRE_EQ(plan.allocation_at(2U)->offset_bytes(), 128U);
  TK_REQUIRE_EQ(plan.allocation_at(3U)->offset_bytes(), 0U);
  TK_REQUIRE_EQ(plan.workspace_bytes(), 192U);
}

TK_TEST("Arena planner coalesces free blocks on both sides") {
  const std::vector<ArenaBufferRequest> requests{
      {64U, 0U, 2U},
      {64U, 0U, 3U},
      {64U, 0U, 2U},
      {192U, 3U, 4U},
  };
  const ArenaPlan plan = require_plan(ArenaPlanner::run(requests));

  TK_REQUIRE_EQ(plan.allocation_at(3U)->offset_bytes(), 0U);
  TK_REQUIRE_EQ(plan.workspace_bytes(), 192U);
}

TK_TEST("Arena planner recoalesces a split block") {
  const std::vector<ArenaBufferRequest> requests{
      {192U, 0U, 2U},
      {64U, 2U, 3U},
      {192U, 3U, 4U},
  };
  const ArenaPlan plan = require_plan(ArenaPlanner::run(requests));

  TK_REQUIRE_EQ(plan.allocation_at(0U)->offset_bytes(), 0U);
  TK_REQUIRE_EQ(plan.allocation_at(1U)->offset_bytes(), 0U);
  TK_REQUIRE_EQ(plan.allocation_at(2U)->offset_bytes(), 0U);
  TK_REQUIRE_EQ(plan.workspace_bytes(), 192U);
}

TK_TEST("Arena planner grows a free block at the workspace frontier") {
  const std::vector<ArenaBufferRequest> requests{
      {64U, 0U, 4U},
      {64U, 0U, 2U},
      {128U, 2U, 3U},
  };
  const ArenaPlan plan = require_plan(
      ArenaPlanner::run(requests, ArenaLimits{3U, 192U}));

  TK_REQUIRE_EQ(plan.allocation_at(0U)->offset_bytes(), 0U);
  TK_REQUIRE_EQ(plan.allocation_at(1U)->offset_bytes(), 64U);
  TK_REQUIRE_EQ(plan.allocation_at(2U)->offset_bytes(), 64U);
  TK_REQUIRE_EQ(plan.workspace_bytes(), 192U);
}

TK_TEST("Arena planner accepts empty requests under zero limits") {
  const ArenaPlan plan = require_plan(ArenaPlanner::run(
      std::span<const ArenaBufferRequest>{}, ArenaLimits{0U, 0U}));
  TK_REQUIRE(plan.allocations().empty());
  TK_REQUIRE_EQ(plan.workspace_bytes(), 0U);
}

TK_TEST("Arena planner reports its exact deterministic workspace limit") {
  const std::vector<ArenaBufferRequest> requests{
      {64U, 0U, 2U},
      {64U, 1U, 3U},
      {64U, 2U, 4U},
  };
  const ArenaPlan exact = require_plan(
      ArenaPlanner::run(requests, ArenaLimits{3U, 128U}));
  TK_REQUIRE_EQ(exact.workspace_bytes(), 128U);

  const auto rejected =
      ArenaPlanner::run(requests, ArenaLimits{3U, 127U});
  const tensorkiln::Diagnostic& error = require_error(
      rejected, ErrorCode::arena_workspace_limit_exceeded);
  TK_REQUIRE_EQ(error.message,
                "arena workspace requires 128 bytes; limit is 127");
}

TK_TEST("Arena planner preserves request validation precedence") {
  const std::vector<ArenaBufferRequest> requests{
      {0U, 3U, 2U},
      {64U, 0U, 1U},
  };
  const auto too_many =
      ArenaPlanner::run(requests, ArenaLimits{1U, 0U});
  TK_REQUIRE_EQ(require_error(too_many,
                             ErrorCode::arena_buffer_limit_exceeded)
                    .message,
                "arena has 2 buffer requests; limit is 1");

  const auto bad_size = ArenaPlanner::run(requests);
  TK_REQUIRE_EQ(
      require_error(bad_size, ErrorCode::arena_buffer_size_invalid).message,
      "arena buffer #b0 has zero payload bytes");
}

}  // namespace
