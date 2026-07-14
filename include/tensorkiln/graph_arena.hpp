#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "tensorkiln/arena.hpp"
#include "tensorkiln/graph.hpp"

namespace tensorkiln {

// Owns a verified storage-only projection of a graph onto one sequential arena.
// Add, MatMul, and Relu nodes receive dense execution steps and compact buffer
// ordinals in source-node order. Input and Constant values remain external.
// Dead compute is retained, while arena-backed graph outputs remain live
// through the final compute step and end at the terminal execution boundary.
//
// For a materialized value produced at step p, its request begins at p and
// ends at the maximum of p + 1, every consumer step + 1, and the execution
// step count when it is a graph output. Thus a consumer's operands overlap its
// result for that step, while reuse at an exclusive lifetime boundary remains
// legal. At every buffer ordinal b, values_by_buffer_ordinal()[b],
// requests()[b], and arena_plan().allocation_at(b) describe the same value.
//
// This result proves byte-placement safety for those lifetimes. It is not an
// executable schedule and makes no layout, view, alias, in-place, prepacking,
// scratch-buffer, allocation, or parallel-execution claim.
//
// The result is move-only. A move, move assignment, or destruction invalidates
// every reference, pointer, and span borrowed from it. After a move, the source
// object supports only assignment or destruction.
class GraphArenaLoweringResult final {
 public:
  GraphArenaLoweringResult(const GraphArenaLoweringResult&) = delete;
  GraphArenaLoweringResult(GraphArenaLoweringResult&&) noexcept = default;
  GraphArenaLoweringResult& operator=(const GraphArenaLoweringResult&) =
      delete;
  GraphArenaLoweringResult& operator=(GraphArenaLoweringResult&&) noexcept =
      default;
  ~GraphArenaLoweringResult() = default;

  [[nodiscard]] std::uint32_t source_node_count() const noexcept {
    return source_node_count_;
  }
  [[nodiscard]] std::uint32_t execution_step_count() const noexcept {
    return execution_step_count_;
  }
  [[nodiscard]] std::span<const ValueId>
  values_by_buffer_ordinal() const noexcept {
    return values_by_buffer_ordinal_;
  }
  [[nodiscard]] std::span<const ArenaBufferRequest> requests() const noexcept {
    return requests_;
  }
  // A null result means only that this artifact has no arena mapping for the
  // handle. It does not classify the handle as external, out of range, or
  // owned by another graph.
  [[nodiscard]] std::optional<std::uint32_t> buffer_ordinal(
      ValueId value) const noexcept;
  [[nodiscard]] const ValueId* value_at(
      std::uint32_t buffer_ordinal) const noexcept;
  [[nodiscard]] const ArenaAllocation* allocation_for(
      ValueId value) const noexcept;
  [[nodiscard]] const ArenaPlan& arena_plan() const noexcept {
    return arena_plan_;
  }
  [[nodiscard]] std::string dump() const;

 private:
  friend class GraphArenaPlacementVerifier;

  GraphArenaLoweringResult(
      std::uint32_t source_node_count,
      std::uint32_t execution_step_count,
      std::vector<ValueId> values_by_buffer_ordinal,
      std::vector<ArenaBufferRequest> requests, ArenaPlan arena_plan,
      std::vector<std::uint32_t> buffer_by_source_ordinal);

  std::uint32_t source_node_count_;
  std::uint32_t execution_step_count_;
  std::vector<ValueId> values_by_buffer_ordinal_;
  std::vector<ArenaBufferRequest> requests_;
  ArenaPlan arena_plan_;
  std::vector<std::uint32_t> buffer_by_source_ordinal_;
};

class GraphArenaPlacementVerifier final {
 public:
  GraphArenaPlacementVerifier() = delete;

  // Reconstructs the storage projection independently and verifies placements
  // against it. Placement #bN addresses the Nth Add, MatMul, or Relu result in
  // source-node order; graph ordinals can differ because Input and Constant
  // nodes do not consume arena buffers.
  //
  // source is borrowed only for this call and must be a valid, non-moved-from
  // VerifiedGraph. The returned artifact owns its mapping and arena plan.
  [[nodiscard]] static Result<GraphArenaLoweringResult> verify(
      const VerifiedGraph& source,
      std::span<const ArenaPlacement> placements,
      ArenaLimits limits = ArenaLimits{});
};

class GraphArenaLowering final {
 public:
  GraphArenaLowering() = delete;

  // Derives requests in a forward pass, applies the deterministic heuristic
  // ArenaPlanner, and then requires an independent reverse reconstruction to
  // accept and exactly match the planned storage projection. Arena resource
  // and representability failures propagate unchanged because no placements
  // exist; invalid derived requests and reverse disagreement are reported as
  // compiler_internal_invariant. Allocation failure remains std::bad_alloc.
  // A successful result proves placement safety, not optimality or execution.
  //
  // source is borrowed only for this call and must be a valid, non-moved-from
  // VerifiedGraph. The returned artifact owns its mapping and arena plan.
  [[nodiscard]] static Result<GraphArenaLoweringResult> run(
      const VerifiedGraph& source,
      ArenaLimits limits = ArenaLimits{});
};

}  // namespace tensorkiln
