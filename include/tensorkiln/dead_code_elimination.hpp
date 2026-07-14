#pragma once

#include <cstdint>
#include <string>

#include "tensorkiln/provenance.hpp"

namespace tensorkiln {

struct DeadCodeEliminationStats final {
  std::uint32_t source_nodes;
  std::uint32_t retained_nodes;
  std::uint32_t removed_nodes;
  std::uint64_t source_constant_elements;
  std::uint64_t retained_constant_elements;
  std::uint64_t removed_constant_elements;

  friend bool operator==(const DeadCodeEliminationStats&,
                         const DeadCodeEliminationStats&) noexcept = default;
};

class DeadCodeEliminationResult final {
 public:
  DeadCodeEliminationResult(const DeadCodeEliminationResult&) = delete;
  DeadCodeEliminationResult(DeadCodeEliminationResult&&) noexcept = default;
  DeadCodeEliminationResult& operator=(
      const DeadCodeEliminationResult&) = delete;
  DeadCodeEliminationResult& operator=(
      DeadCodeEliminationResult&&) noexcept = default;
  ~DeadCodeEliminationResult() = default;

  [[nodiscard]] const VerifiedGraph& graph() const noexcept { return graph_; }
  [[nodiscard]] const GraphProvenance& provenance() const noexcept {
    return provenance_;
  }
  [[nodiscard]] const DeadCodeEliminationStats& stats() const noexcept {
    return stats_;
  }
  [[nodiscard]] std::string dump() const;

 private:
  friend class DeadCodeElimination;

  DeadCodeEliminationResult(VerifiedGraph graph,
                            GraphProvenance provenance,
                            DeadCodeEliminationStats stats);

  VerifiedGraph graph_;
  GraphProvenance provenance_;
  DeadCodeEliminationStats stats_;
};

class DeadCodeElimination final {
 public:
  DeadCodeElimination() = delete;

  [[nodiscard]] static Result<DeadCodeEliminationResult> run(
      const VerifiedGraph& source);

  [[nodiscard]] static Result<DeadCodeEliminationResult> run(
      const VerifiedGraph& source,
      const GraphProvenance& source_provenance);
};

}  // namespace tensorkiln
