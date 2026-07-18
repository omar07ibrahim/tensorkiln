#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "tensorkiln/graph_arena.hpp"

namespace tensorkiln {

inline constexpr std::uint32_t kDefaultMaxPlanValues = 4096U;
inline constexpr std::uint32_t kDefaultMaxPlanSteps = 4096U;
inline constexpr std::uint32_t kDefaultMaxPlanOutputs = 64U;
inline constexpr std::uint64_t kDefaultMaxPlanConstantBytes =
    UINT64_C(1) << 28U;
inline constexpr std::uint64_t kDefaultMaxPlanScalarSteps =
    UINT64_C(1) << 30U;

struct ExecutionPlanLimits final {
  std::uint32_t max_values = kDefaultMaxPlanValues;
  std::uint32_t max_steps = kDefaultMaxPlanSteps;
  std::uint32_t max_outputs = kDefaultMaxPlanOutputs;
  std::uint64_t max_owned_constant_bytes =
      kDefaultMaxPlanConstantBytes;
  std::uint64_t max_scalar_steps = kDefaultMaxPlanScalarSteps;
  ArenaLimits arena_limits{};

  friend bool operator==(const ExecutionPlanLimits&,
                         const ExecutionPlanLimits&) noexcept = default;
};

struct ExecutionPlanStats final {
  std::uint32_t value_count;
  std::uint32_t input_count;
  std::uint32_t constant_count;
  std::uint32_t step_count;
  std::uint32_t output_count;
  std::uint64_t owned_constant_bytes;
  std::uint64_t scalar_steps;
  std::uint64_t workspace_bytes;

  friend bool operator==(const ExecutionPlanStats&,
                         const ExecutionPlanStats&) noexcept = default;
};

enum class PlanStorageKind : std::uint8_t {
  input,
  constant,
  arena,
};

[[nodiscard]] std::string_view plan_storage_kind_name(
    PlanStorageKind kind) noexcept;

enum class DenseKernelKind : std::uint8_t {
  add_contiguous_f32,
  add_broadcast_f32,
  matmul_rank2_f32,
  matmul_batched_f32,
  relu_contiguous_f32,
};

[[nodiscard]] std::string_view dense_kernel_kind_name(
    DenseKernelKind kind) noexcept;

class DenseLayout final {
 public:
  [[nodiscard]] std::size_t rank() const noexcept { return rank_; }
  [[nodiscard]] std::span<const std::uint64_t> strides_elements()
      const noexcept {
    return std::span<const std::uint64_t>(strides_elements_.data(), rank_);
  }
  [[nodiscard]] std::uint64_t elements() const noexcept { return elements_; }

  friend bool operator==(const DenseLayout&, const DenseLayout&) noexcept =
      default;

 private:
  friend class ExecutionPlanVerifier;

  DenseLayout(std::array<std::uint64_t, kMaxRank> strides_elements,
              std::size_t rank, std::uint64_t elements) noexcept;

  std::array<std::uint64_t, kMaxRank> strides_elements_{};
  std::size_t rank_;
  std::uint64_t elements_;
};

class PlanStorage final {
 public:
  [[nodiscard]] PlanStorageKind kind() const noexcept { return kind_; }
  // Input and Constant storage use a dense class-local ordinal. Arena storage
  // uses the arena buffer ordinal. Only arena storage may have a non-zero
  // byte offset.
  [[nodiscard]] std::uint32_t ordinal() const noexcept { return ordinal_; }
  [[nodiscard]] std::uint64_t offset_bytes() const noexcept {
    return offset_bytes_;
  }

  friend bool operator==(const PlanStorage&, const PlanStorage&) noexcept =
      default;

 private:
  friend class ExecutionPlanVerifier;

  PlanStorage(PlanStorageKind kind, std::uint32_t ordinal,
              std::uint64_t offset_bytes) noexcept;

  PlanStorageKind kind_;
  std::uint32_t ordinal_;
  std::uint64_t offset_bytes_;
};

class PlanValue final {
 public:
  [[nodiscard]] ValueId source_value() const noexcept {
    return source_value_;
  }
  [[nodiscard]] const TensorType& type() const noexcept { return type_; }
  [[nodiscard]] const DenseLayout& layout() const noexcept { return layout_; }
  [[nodiscard]] const PlanStorage& storage() const noexcept {
    return storage_;
  }

 private:
  friend class ExecutionPlanVerifier;

  PlanValue(ValueId source_value, TensorType type, DenseLayout layout,
            PlanStorage storage);

  ValueId source_value_;
  TensorType type_;
  DenseLayout layout_;
  PlanStorage storage_;
};

class ExecutionStep final {
 public:
  [[nodiscard]] std::uint32_t ordinal() const noexcept { return ordinal_; }
  [[nodiscard]] NodeId source_node() const noexcept { return source_node_; }
  [[nodiscard]] DenseKernelKind kernel() const noexcept { return kernel_; }
  [[nodiscard]] std::span<const ValueId> operands() const noexcept {
    return operands_;
  }
  [[nodiscard]] ValueId output() const noexcept { return output_; }
  [[nodiscard]] std::uint64_t scalar_steps() const noexcept {
    return scalar_steps_;
  }

 private:
  friend class ExecutionPlanVerifier;

  ExecutionStep(std::uint32_t ordinal, NodeId source_node,
                DenseKernelKind kernel, std::vector<ValueId> operands,
                ValueId output, std::uint64_t scalar_steps);

  std::uint32_t ordinal_;
  NodeId source_node_;
  DenseKernelKind kernel_;
  std::vector<ValueId> operands_;
  ValueId output_;
  std::uint64_t scalar_steps_;
};

struct ExecutionStepSpec final {
  std::uint32_t source_node_ordinal;
  DenseKernelKind kernel;

  friend bool operator==(const ExecutionStepSpec&,
                         const ExecutionStepSpec&) noexcept = default;
};

// The candidate deliberately carries only decisions: step-to-kernel choices
// and arena offsets. The verifier reconstructs operands, layouts, storage,
// lifetimes, work accounting, and all other plan facts from the source graph.
struct ExecutionPlanCandidate final {
  std::span<const ExecutionStepSpec> steps;
  std::span<const ArenaPlacement> placements;
};

// Owns the source graph, verified dense plan records, and verified arena
// projection. Borrowed references and spans are invalidated by move,
// move-assignment, or destruction. A moved-from object supports only
// assignment or destruction.
class ExecutionPlan final {
 public:
  ExecutionPlan(const ExecutionPlan&) = delete;
  ExecutionPlan(ExecutionPlan&&) noexcept = default;
  ExecutionPlan& operator=(const ExecutionPlan&) = delete;
  ExecutionPlan& operator=(ExecutionPlan&&) noexcept = default;
  ~ExecutionPlan() = default;

  [[nodiscard]] const ExecutionPlanLimits& limits() const noexcept {
    return limits_;
  }
  [[nodiscard]] const ExecutionPlanStats& stats() const noexcept {
    return stats_;
  }
  [[nodiscard]] const VerifiedGraph& graph() const noexcept { return graph_; }
  [[nodiscard]] const ArenaPlan& arena_plan() const noexcept {
    return arena_.arena_plan();
  }
  [[nodiscard]] std::span<const PlanValue> values() const noexcept {
    return values_;
  }
  [[nodiscard]] std::span<const ExecutionStep> steps() const noexcept {
    return steps_;
  }
  // A null result means the handle is foreign or outside this plan's dense
  // source-value domain.
  [[nodiscard]] const PlanValue* value(ValueId source_value) const noexcept;
  [[nodiscard]] std::string dump() const;

 private:
  friend class ExecutionPlanVerifier;

  ExecutionPlan(ExecutionPlanLimits limits, ExecutionPlanStats stats,
                VerifiedGraph graph, GraphArenaLoweringResult arena,
                std::vector<PlanValue> values,
                std::vector<ExecutionStep> steps);

  ExecutionPlanLimits limits_;
  ExecutionPlanStats stats_;
  VerifiedGraph graph_;
  GraphArenaLoweringResult arena_;
  std::vector<PlanValue> values_;
  std::vector<ExecutionStep> steps_;
};

class ExecutionPlanVerifier final {
 public:
  ExecutionPlanVerifier() = delete;

  // source is borrowed only for this call and must be a valid,
  // non-moved-from VerifiedGraph. The returned plan owns its graph copy and
  // all verified plan metadata.
  [[nodiscard]] static Result<ExecutionPlan> verify(
      const VerifiedGraph& source, const ExecutionPlanCandidate& candidate,
      ExecutionPlanLimits limits = ExecutionPlanLimits{});
};

class ExecutionPlanCompiler final {
 public:
  ExecutionPlanCompiler() = delete;

  // source is borrowed only for this call and must be a valid,
  // non-moved-from VerifiedGraph. The returned plan owns its graph copy and
  // all verified plan metadata.
  [[nodiscard]] static Result<ExecutionPlan> run(
      const VerifiedGraph& source,
      ExecutionPlanLimits limits = ExecutionPlanLimits{});
};

}  // namespace tensorkiln
