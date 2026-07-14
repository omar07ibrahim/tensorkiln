#pragma once

#include <cstdint>
#include <string>

#include "tensorkiln/provenance.hpp"

namespace tensorkiln {

struct StructuralCanonicalizationStats final {
  std::uint32_t source_nodes;
  std::uint32_t result_nodes;
  std::uint32_t merged_nodes;
  std::uint32_t common_subexpressions;
  std::uint32_t redundant_relus;
  std::uint32_t preserved_output_distinctions;

  friend bool operator==(const StructuralCanonicalizationStats&,
                         const StructuralCanonicalizationStats&) noexcept =
      default;
};

class StructuralCanonicalizationResult final {
 public:
  StructuralCanonicalizationResult(
      const StructuralCanonicalizationResult&) = delete;
  StructuralCanonicalizationResult(
      StructuralCanonicalizationResult&&) noexcept = default;
  StructuralCanonicalizationResult& operator=(
      const StructuralCanonicalizationResult&) = delete;
  StructuralCanonicalizationResult& operator=(
      StructuralCanonicalizationResult&&) noexcept = default;
  ~StructuralCanonicalizationResult() = default;

  [[nodiscard]] const VerifiedGraph& graph() const noexcept { return graph_; }
  [[nodiscard]] const GraphProvenance& provenance() const noexcept {
    return provenance_;
  }
  [[nodiscard]] const StructuralCanonicalizationStats& stats() const noexcept {
    return stats_;
  }
  [[nodiscard]] std::string dump() const;

 private:
  friend class StructuralCanonicalization;

  StructuralCanonicalizationResult(
      VerifiedGraph graph, GraphProvenance provenance,
      StructuralCanonicalizationStats stats);

  VerifiedGraph graph_;
  GraphProvenance provenance_;
  StructuralCanonicalizationStats stats_;
};

class StructuralCanonicalization final {
 public:
  StructuralCanonicalization() = delete;

  [[nodiscard]] static Result<StructuralCanonicalizationResult> run(
      const VerifiedGraph& source);

  [[nodiscard]] static Result<StructuralCanonicalizationResult> run(
      const VerifiedGraph& source,
      const GraphProvenance& source_provenance);
};

}  // namespace tensorkiln
