#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "tensorkiln/result.hpp"

namespace tensorkiln {

inline constexpr std::uint64_t kArenaAlignmentBytes = 64U;
inline constexpr std::uint32_t kDefaultMaxArenaBuffers = 4096U;
inline constexpr std::uint64_t kDefaultMaxArenaWorkspaceBytes =
    UINT64_C(1) << 28U;

struct ArenaLimits final {
  std::uint32_t max_buffers = kDefaultMaxArenaBuffers;
  std::uint64_t max_workspace_bytes = kDefaultMaxArenaWorkspaceBytes;

  friend bool operator==(const ArenaLimits&, const ArenaLimits&) noexcept =
      default;
};

struct ArenaBufferRequest final {
  std::uint64_t size_bytes;
  std::uint32_t live_begin_step;
  std::uint32_t live_end_step_exclusive;

  friend bool operator==(const ArenaBufferRequest&,
                         const ArenaBufferRequest&) noexcept = default;
};

struct ArenaPlacement final {
  std::uint32_t buffer_ordinal;
  std::uint64_t offset_bytes;

  friend bool operator==(const ArenaPlacement&,
                         const ArenaPlacement&) noexcept = default;
};

struct ArenaPlanStats final {
  std::uint32_t buffer_count;
  std::uint64_t total_payload_bytes;
  std::uint64_t total_reserved_bytes;
  std::uint64_t peak_live_reserved_bytes;
  std::uint64_t workspace_bytes;

  friend bool operator==(const ArenaPlanStats&,
                         const ArenaPlanStats&) noexcept = default;
};

class ArenaAllocation final {
 public:
  [[nodiscard]] std::uint32_t buffer_ordinal() const noexcept {
    return buffer_ordinal_;
  }
  [[nodiscard]] std::uint64_t offset_bytes() const noexcept {
    return offset_bytes_;
  }
  [[nodiscard]] std::uint64_t payload_bytes() const noexcept {
    return payload_bytes_;
  }
  [[nodiscard]] std::uint64_t reserved_bytes() const noexcept {
    return reserved_bytes_;
  }
  [[nodiscard]] std::uint32_t live_begin_step() const noexcept {
    return live_begin_step_;
  }
  [[nodiscard]] std::uint32_t live_end_step_exclusive() const noexcept {
    return live_end_step_exclusive_;
  }

 private:
  friend class ArenaPlacementVerifier;

  ArenaAllocation(std::uint32_t buffer_ordinal, std::uint64_t offset_bytes,
                  std::uint64_t payload_bytes,
                  std::uint64_t reserved_bytes,
                  std::uint32_t live_begin_step,
                  std::uint32_t live_end_step_exclusive) noexcept;

  std::uint32_t buffer_ordinal_;
  std::uint64_t offset_bytes_;
  std::uint64_t payload_bytes_;
  std::uint64_t reserved_bytes_;
  std::uint32_t live_begin_step_;
  std::uint32_t live_end_step_exclusive_;
};

class ArenaPlan final {
 public:
  ArenaPlan(const ArenaPlan&) = default;
  ArenaPlan(ArenaPlan&&) noexcept = default;
  ArenaPlan& operator=(const ArenaPlan&) = default;
  ArenaPlan& operator=(ArenaPlan&&) noexcept = default;
  ~ArenaPlan() = default;

  [[nodiscard]] std::span<const ArenaAllocation> allocations() const noexcept {
    return allocations_;
  }
  [[nodiscard]] const ArenaAllocation* allocation_at(
      std::uint32_t buffer_ordinal) const noexcept;
  [[nodiscard]] const ArenaLimits& limits() const noexcept { return limits_; }
  [[nodiscard]] const ArenaPlanStats& stats() const noexcept { return stats_; }
  [[nodiscard]] std::uint64_t workspace_bytes() const noexcept {
    return stats_.workspace_bytes;
  }
  [[nodiscard]] std::string dump() const;

 private:
  friend class ArenaPlacementVerifier;

  ArenaPlan(ArenaLimits limits, ArenaPlanStats stats,
            std::vector<ArenaAllocation> allocations);

  ArenaLimits limits_;
  ArenaPlanStats stats_;
  std::vector<ArenaAllocation> allocations_;
};

class ArenaPlacementVerifier final {
 public:
  ArenaPlacementVerifier() = delete;

  [[nodiscard]] static Result<ArenaPlan> verify(
      std::span<const ArenaBufferRequest> requests,
      std::span<const ArenaPlacement> placements,
      ArenaLimits limits = ArenaLimits{});
};

class ArenaPlanner final {
 public:
  ArenaPlanner() = delete;

  [[nodiscard]] static Result<ArenaPlan> run(
      std::span<const ArenaBufferRequest> requests,
      ArenaLimits limits = ArenaLimits{});
};

}  // namespace tensorkiln
