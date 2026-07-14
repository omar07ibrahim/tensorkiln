#include "test.hpp"

#include <array>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "tensorkiln/arena.hpp"

namespace {

using tensorkiln::ArenaBufferRequest;
using tensorkiln::ArenaLimits;
using tensorkiln::ArenaPlacement;
using tensorkiln::ArenaPlacementVerifier;
using tensorkiln::ArenaPlan;
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

TK_TEST("Arena verifier canonicalizes placements and derives exact statistics") {
  const std::vector<ArenaBufferRequest> requests{
      {24U, 0U, 3U},
      {40U, 1U, 2U},
      {32U, 2U, 4U},
      {64U, 3U, 5U},
  };
  const std::vector<ArenaPlacement> placements{
      {3U, 0U},
      {1U, 64U},
      {0U, 0U},
      {2U, 64U},
  };

  const ArenaPlan plan = require_plan(
      ArenaPlacementVerifier::verify(requests, placements));
  TK_REQUIRE_EQ(plan.allocations().size(), 4U);
  for (std::uint32_t ordinal = 0U; ordinal < 4U; ++ordinal) {
    TK_REQUIRE(plan.allocation_at(ordinal) != nullptr);
    TK_REQUIRE_EQ(plan.allocation_at(ordinal)->buffer_ordinal(), ordinal);
  }
  TK_REQUIRE(plan.allocation_at(4U) == nullptr);
  TK_REQUIRE_EQ(plan.allocation_at(0U)->offset_bytes(), 0U);
  TK_REQUIRE_EQ(plan.allocation_at(1U)->offset_bytes(), 64U);
  TK_REQUIRE_EQ(plan.allocation_at(2U)->offset_bytes(), 64U);
  TK_REQUIRE_EQ(plan.allocation_at(3U)->offset_bytes(), 0U);
  TK_REQUIRE_EQ(plan.allocation_at(0U)->payload_bytes(), 24U);
  TK_REQUIRE_EQ(plan.allocation_at(0U)->reserved_bytes(), 64U);

  const tensorkiln::ArenaPlanStats expected_stats{
      4U,
      160U,
      256U,
      128U,
      128U,
  };
  TK_REQUIRE_EQ(plan.stats(), expected_stats);
  TK_REQUIRE_EQ(plan.workspace_bytes(), 128U);

  const std::string expected_dump =
      "tensorkiln.arena_plan v0 {\n"
      "  alignment_bytes=64\n"
      "  limits {buffers=4096, workspace_bytes=268435456}\n"
      "  stats {buffers=4, payload_bytes=160, reserved_bytes=256, "
      "peak_live_reserved_bytes=128, workspace_bytes=128}\n"
      "  #b0 offset=0 payload=24 reserved=64 live=[0,3)\n"
      "  #b1 offset=64 payload=40 reserved=64 live=[1,2)\n"
      "  #b2 offset=64 payload=32 reserved=64 live=[2,4)\n"
      "  #b3 offset=0 payload=64 reserved=64 live=[3,5)\n"
      "}\n";
  TK_REQUIRE_EQ(plan.dump(), expected_dump);
  TK_REQUIRE_EQ(plan.dump(), plan.dump());
}

TK_TEST("Arena verifier accepts the empty zero-limit plan") {
  const ArenaLimits limits{0U, 0U};
  const ArenaPlan plan = require_plan(ArenaPlacementVerifier::verify(
      std::span<const ArenaBufferRequest>{},
      std::span<const ArenaPlacement>{}, limits));

  TK_REQUIRE(plan.allocations().empty());
  TK_REQUIRE_EQ(plan.limits(), limits);
  TK_REQUIRE_EQ(plan.workspace_bytes(), 0U);
  TK_REQUIRE_EQ(plan.stats().peak_live_reserved_bytes, 0U);
}

TK_TEST("Arena verifier rounds payloads to exact 64-byte reservations") {
  const std::vector<ArenaBufferRequest> requests{
      {1U, 0U, 1U},
      {63U, 0U, 1U},
      {64U, 0U, 1U},
      {65U, 0U, 1U},
  };
  const std::vector<ArenaPlacement> placements{
      {0U, 0U},
      {1U, 64U},
      {2U, 128U},
      {3U, 192U},
  };
  const ArenaPlan plan = require_plan(
      ArenaPlacementVerifier::verify(requests, placements));

  TK_REQUIRE_EQ(plan.allocation_at(0U)->reserved_bytes(), 64U);
  TK_REQUIRE_EQ(plan.allocation_at(1U)->reserved_bytes(), 64U);
  TK_REQUIRE_EQ(plan.allocation_at(2U)->reserved_bytes(), 64U);
  TK_REQUIRE_EQ(plan.allocation_at(3U)->reserved_bytes(), 128U);
  TK_REQUIRE_EQ(plan.workspace_bytes(), 320U);
  TK_REQUIRE_EQ(plan.stats().peak_live_reserved_bytes, 320U);
}

TK_TEST("Arena verifier permits reuse exactly at a lifetime boundary") {
  const std::vector<ArenaBufferRequest> requests{
      {64U, 0U, 2U},
      {64U, 2U, 4U},
  };
  const std::vector<ArenaPlacement> placements{
      {0U, 0U},
      {1U, 0U},
  };
  const ArenaPlan plan = require_plan(
      ArenaPlacementVerifier::verify(requests, placements));

  TK_REQUIRE_EQ(plan.workspace_bytes(), 64U);
  TK_REQUIRE_EQ(plan.stats().peak_live_reserved_bytes, 64U);
}

TK_TEST("Arena verifier rejects buffers that are live on the same step") {
  const std::vector<ArenaBufferRequest> requests{
      {64U, 0U, 2U},
      {64U, 1U, 3U},
  };
  const std::vector<ArenaPlacement> placements{
      {0U, 0U},
      {1U, 0U},
  };
  const auto verified = ArenaPlacementVerifier::verify(requests, placements);
  const tensorkiln::Diagnostic& error =
      require_error(verified, ErrorCode::arena_live_overlap);
  TK_REQUIRE_EQ(
      error.message,
      "arena buffers #b0 and #b1 overlap in bytes [0,64) and [0,64) "
      "while lifetimes [0,2) and [1,3) intersect");
}

TK_TEST("Arena verifier accepts adjacent live byte ranges") {
  const std::vector<ArenaBufferRequest> requests{
      {64U, 0U, 2U},
      {64U, 0U, 2U},
  };
  const std::vector<ArenaPlacement> placements{
      {0U, 0U},
      {1U, 64U},
  };
  const ArenaPlan plan = require_plan(
      ArenaPlacementVerifier::verify(requests, placements));
  TK_REQUIRE_EQ(plan.workspace_bytes(), 128U);
}

TK_TEST("Arena verifier checks predecessor and successor byte overlaps") {
  const std::vector<ArenaBufferRequest> predecessor_requests{
      {65U, 0U, 3U},
      {64U, 1U, 2U},
  };
  const std::vector<ArenaPlacement> predecessor_placements{
      {0U, 0U},
      {1U, 64U},
  };
  const auto predecessor = ArenaPlacementVerifier::verify(
      predecessor_requests, predecessor_placements);
  TK_REQUIRE_EQ(require_error(predecessor, ErrorCode::arena_live_overlap)
                    .message,
                "arena buffers #b0 and #b1 overlap in bytes [0,128) and "
                "[64,128) while lifetimes [0,3) and [1,2) intersect");

  const std::vector<ArenaBufferRequest> successor_requests{
      {64U, 0U, 3U},
      {65U, 1U, 2U},
  };
  const std::vector<ArenaPlacement> successor_placements{
      {0U, 64U},
      {1U, 0U},
  };
  const auto successor = ArenaPlacementVerifier::verify(
      successor_requests, successor_placements);
  TK_REQUIRE_EQ(require_error(successor, ErrorCode::arena_live_overlap)
                    .message,
                "arena buffers #b0 and #b1 overlap in bytes [64,128) and "
                "[0,128) while lifetimes [0,3) and [1,2) intersect");
}

TK_TEST("Arena verifier accepts a live allocation in an exact byte gap") {
  const std::vector<ArenaBufferRequest> requests{
      {64U, 0U, 3U},
      {64U, 0U, 3U},
      {64U, 1U, 2U},
  };
  const std::vector<ArenaPlacement> placements{
      {0U, 0U},
      {1U, 128U},
      {2U, 64U},
  };
  const ArenaPlan plan = require_plan(
      ArenaPlacementVerifier::verify(requests, placements));
  TK_REQUIRE_EQ(plan.workspace_bytes(), 192U);
  TK_REQUIRE_EQ(plan.stats().peak_live_reserved_bytes, 192U);
}

TK_TEST("Arena verifier expires every buffer at a shared boundary") {
  const std::vector<ArenaBufferRequest> requests{
      {64U, 0U, 2U},
      {64U, 0U, 2U},
      {64U, 2U, 4U},
      {64U, 2U, 4U},
  };
  const std::vector<ArenaPlacement> placements{
      {0U, 0U},
      {1U, 64U},
      {2U, 0U},
      {3U, 64U},
  };
  const ArenaPlan plan = require_plan(
      ArenaPlacementVerifier::verify(requests, placements));
  TK_REQUIRE_EQ(plan.workspace_bytes(), 128U);
  TK_REQUIRE_EQ(plan.stats().peak_live_reserved_bytes, 128U);
}

TK_TEST("Arena verifier permits a valid workspace with leading gaps") {
  const std::vector<ArenaBufferRequest> requests{{64U, 0U, 1U}};
  const std::vector<ArenaPlacement> placements{{0U, 128U}};
  const ArenaPlan plan = require_plan(ArenaPlacementVerifier::verify(
      requests, placements, ArenaLimits{1U, 192U}));

  TK_REQUIRE_EQ(plan.stats().total_reserved_bytes, 64U);
  TK_REQUIRE_EQ(plan.stats().peak_live_reserved_bytes, 64U);
  TK_REQUIRE_EQ(plan.workspace_bytes(), 192U);
}

TK_TEST("Arena verifier applies the exact inclusive workspace limit") {
  const std::vector<ArenaBufferRequest> requests{{65U, 0U, 1U}};
  const std::vector<ArenaPlacement> placements{{0U, 0U}};

  const ArenaPlan exact = require_plan(ArenaPlacementVerifier::verify(
      requests, placements, ArenaLimits{1U, 128U}));
  TK_REQUIRE_EQ(exact.workspace_bytes(), 128U);

  const auto rejected = ArenaPlacementVerifier::verify(
      requests, placements, ArenaLimits{1U, 127U});
  const tensorkiln::Diagnostic& error = require_error(
      rejected, ErrorCode::arena_workspace_limit_exceeded);
  TK_REQUIRE_EQ(error.message,
                "arena workspace requires 128 bytes; limit is 127");
}

TK_TEST("Arena verifier rejects request count before request contents") {
  const std::vector<ArenaBufferRequest> requests{
      {0U, 3U, 2U},
      {64U, 0U, 1U},
  };
  const auto verified = ArenaPlacementVerifier::verify(
      requests, std::span<const ArenaPlacement>{}, ArenaLimits{1U, 0U});
  const tensorkiln::Diagnostic& error = require_error(
      verified, ErrorCode::arena_buffer_limit_exceeded);
  TK_REQUIRE_EQ(error.message,
                "arena has 2 buffer requests; limit is 1");
}

TK_TEST("Arena verifier rejects zero payload before a bad lifetime") {
  const std::vector<ArenaBufferRequest> requests{{0U, 3U, 2U}};
  const auto verified = ArenaPlacementVerifier::verify(
      requests, std::span<const ArenaPlacement>{});
  const tensorkiln::Diagnostic& error = require_error(
      verified, ErrorCode::arena_buffer_size_invalid);
  TK_REQUIRE_EQ(error.message,
                "arena buffer #b0 has zero payload bytes");
}

TK_TEST("Arena verifier rejects an empty or reversed lifetime") {
  const std::vector<ArenaBufferRequest> empty{{64U, 7U, 7U}};
  const auto empty_result = ArenaPlacementVerifier::verify(
      empty, std::span<const ArenaPlacement>{});
  TK_REQUIRE_EQ(require_error(empty_result, ErrorCode::arena_lifetime_invalid)
                    .message,
                "arena buffer #b0 has invalid lifetime [7,7); begin must be "
                "less than end");

  const std::vector<ArenaBufferRequest> reversed{{64U, 8U, 7U}};
  const auto reversed_result = ArenaPlacementVerifier::verify(
      reversed, std::span<const ArenaPlacement>{});
  TK_REQUIRE_EQ(
      require_error(reversed_result, ErrorCode::arena_lifetime_invalid)
          .message,
      "arena buffer #b0 has invalid lifetime [8,7); begin must be less than "
      "end");
}

TK_TEST("Arena verifier rejects reservation and aggregate size overflow") {
  const std::vector<ArenaBufferRequest> alignment_overflow{
      {std::numeric_limits<std::uint64_t>::max(), 0U, 1U},
  };
  const auto bad_alignment = ArenaPlacementVerifier::verify(
      alignment_overflow, std::span<const ArenaPlacement>{});
  TK_REQUIRE_EQ(require_error(bad_alignment, ErrorCode::arena_size_overflow)
                    .message,
                "arena buffer #b0 payload size 18446744073709551615 cannot "
                "be rounded to 64-byte alignment");

  const std::vector<ArenaBufferRequest> aggregate_overflow{
      {std::numeric_limits<std::uint64_t>::max() - 63U, 0U, 1U},
      {64U, 1U, 2U},
  };
  const auto bad_total = ArenaPlacementVerifier::verify(
      aggregate_overflow, std::span<const ArenaPlacement>{});
  TK_REQUIRE_EQ(require_error(bad_total, ErrorCode::arena_size_overflow)
                    .message,
                "arena payload total overflows uint64 at buffer #b1");

  const std::vector<ArenaBufferRequest> reserved_total_overflow{
      {std::numeric_limits<std::uint64_t>::max() - 127U, 0U, 1U},
      {127U, 1U, 2U},
  };
  const auto bad_reserved_total = ArenaPlacementVerifier::verify(
      reserved_total_overflow, std::span<const ArenaPlacement>{});
  TK_REQUIRE_EQ(
      require_error(bad_reserved_total, ErrorCode::arena_size_overflow)
          .message,
      "arena reserved-byte total overflows uint64 at buffer #b1");
}

TK_TEST("Arena verifier rejects a placement count mismatch first") {
  const std::vector<ArenaBufferRequest> requests{{64U, 0U, 1U}};
  const std::vector<ArenaPlacement> placements{
      {9U, 1U},
      {0U, 0U},
  };
  const auto verified = ArenaPlacementVerifier::verify(requests, placements);
  const tensorkiln::Diagnostic& error = require_error(
      verified, ErrorCode::arena_placement_count_mismatch);
  TK_REQUIRE_EQ(error.message, "arena plan has 2 placements; expected 1");
}

TK_TEST("Arena verifier rejects unknown duplicate and unaligned placements") {
  const std::vector<ArenaBufferRequest> one_request{{64U, 0U, 1U}};
  const std::vector<ArenaPlacement> unknown{{7U, 1U}};
  const auto bad_unknown =
      ArenaPlacementVerifier::verify(one_request, unknown);
  TK_REQUIRE_EQ(
      require_error(bad_unknown,
                    ErrorCode::arena_placement_buffer_not_found)
          .message,
      "arena placement 0 references unknown buffer #b7");

  const std::vector<ArenaBufferRequest> two_requests{
      {64U, 0U, 1U},
      {64U, 1U, 2U},
  };
  const std::vector<ArenaPlacement> duplicate{
      {0U, 0U},
      {0U, 1U},
  };
  const auto bad_duplicate =
      ArenaPlacementVerifier::verify(two_requests, duplicate);
  TK_REQUIRE_EQ(
      require_error(bad_duplicate, ErrorCode::arena_placement_duplicate)
          .message,
      "arena buffer #b0 has multiple placements");

  const std::vector<ArenaPlacement> unaligned{{0U, 1U}};
  const auto bad_alignment =
      ArenaPlacementVerifier::verify(one_request, unaligned);
  TK_REQUIRE_EQ(
      require_error(bad_alignment, ErrorCode::arena_alignment_invalid)
          .message,
      "arena buffer #b0 offset 1 is not 64-byte aligned");
}

TK_TEST("Arena verifier rejects an overflowing placed byte range") {
  const std::vector<ArenaBufferRequest> requests{{65U, 0U, 1U}};
  const std::vector<ArenaPlacement> placements{
      {0U, std::numeric_limits<std::uint64_t>::max() - 63U},
  };
  const auto verified = ArenaPlacementVerifier::verify(requests, placements);
  const tensorkiln::Diagnostic& error =
      require_error(verified, ErrorCode::arena_size_overflow);
  TK_REQUIRE_EQ(
      error.message,
      "arena buffer #b0 offset 18446744073709551552 plus reserved size 128 "
      "overflows uint64");
}

TK_TEST("Arena verifier reports overlap before workspace policy failure") {
  const std::vector<ArenaBufferRequest> requests{
      {64U, 0U, 2U},
      {64U, 1U, 3U},
  };
  const std::vector<ArenaPlacement> placements{
      {0U, 0U},
      {1U, 0U},
  };
  const auto verified = ArenaPlacementVerifier::verify(
      requests, placements, ArenaLimits{2U, 0U});
  TK_REQUIRE_EQ(require_error(verified, ErrorCode::arena_live_overlap).code,
                ErrorCode::arena_live_overlap);
}

TK_TEST("Arena verifier reports stable public error names") {
  constexpr std::array<std::pair<ErrorCode, std::string_view>, 11U> cases{{
      {ErrorCode::arena_buffer_limit_exceeded,
       "arena_buffer_limit_exceeded"},
      {ErrorCode::arena_buffer_size_invalid, "arena_buffer_size_invalid"},
      {ErrorCode::arena_lifetime_invalid, "arena_lifetime_invalid"},
      {ErrorCode::arena_placement_count_mismatch,
       "arena_placement_count_mismatch"},
      {ErrorCode::arena_placement_buffer_not_found,
       "arena_placement_buffer_not_found"},
      {ErrorCode::arena_placement_duplicate, "arena_placement_duplicate"},
      {ErrorCode::arena_alignment_invalid, "arena_alignment_invalid"},
      {ErrorCode::arena_size_overflow, "arena_size_overflow"},
      {ErrorCode::arena_live_overlap, "arena_live_overlap"},
      {ErrorCode::arena_workspace_limit_exceeded,
       "arena_workspace_limit_exceeded"},
      {ErrorCode::arena_workspace_unaddressable,
       "arena_workspace_unaddressable"},
  }};
  for (const auto& [code, name] : cases) {
    TK_REQUIRE_EQ(tensorkiln::error_code_name(code), name);
  }
}

}  // namespace
