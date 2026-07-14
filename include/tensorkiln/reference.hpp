#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "tensorkiln/graph.hpp"

namespace tensorkiln {

inline constexpr std::uint64_t kDefaultMaxReferenceMaterializedBytes =
    UINT64_C(1) << 28U;
inline constexpr std::uint64_t kDefaultMaxReferenceScalarSteps =
    UINT64_C(1) << 28U;

struct InputBinding final {
  std::string_view name;
  std::span<const float> data;
};

struct ReferenceLimits final {
  std::uint64_t max_materialized_bytes =
      kDefaultMaxReferenceMaterializedBytes;
  std::uint64_t max_scalar_steps = kDefaultMaxReferenceScalarSteps;
};

class Tensor final {
 public:
  Tensor(const Tensor&) = delete;
  Tensor(Tensor&&) noexcept = default;
  Tensor& operator=(const Tensor&) = delete;
  Tensor& operator=(Tensor&&) noexcept = default;
  ~Tensor() = default;

  [[nodiscard]] const TensorType& type() const noexcept { return type_; }
  [[nodiscard]] std::span<const float> data() const noexcept { return data_; }

 private:
  friend class ReferenceInterpreter;

  Tensor(TensorType type, std::vector<float> data);

  TensorType type_;
  std::vector<float> data_;
};

class ReferenceResult final {
 public:
  ReferenceResult(const ReferenceResult&) = delete;
  ReferenceResult(ReferenceResult&&) noexcept = default;
  ReferenceResult& operator=(const ReferenceResult&) = delete;
  ReferenceResult& operator=(ReferenceResult&&) noexcept = default;
  ~ReferenceResult() = default;

  [[nodiscard]] const Tensor* value(ValueId id) const noexcept;
  [[nodiscard]] const Tensor* output(std::string_view name) const noexcept;
  [[nodiscard]] std::uint64_t materialized_bytes() const noexcept {
    return materialized_bytes_;
  }
  [[nodiscard]] std::uint64_t scalar_steps() const noexcept {
    return scalar_steps_;
  }

 private:
  friend class ReferenceInterpreter;

  struct OutputRecord final {
    std::string name;
    ValueId value;
  };

  ReferenceResult(std::vector<ValueId> value_ids, std::vector<Tensor> values,
                  std::vector<OutputRecord> outputs,
                  std::uint64_t materialized_bytes,
                  std::uint64_t scalar_steps);

  std::vector<ValueId> value_ids_;
  std::vector<Tensor> values_;
  std::vector<OutputRecord> outputs_;
  std::uint64_t materialized_bytes_;
  std::uint64_t scalar_steps_;
};

class ReferenceInterpreter final {
 public:
  ReferenceInterpreter() = delete;

  [[nodiscard]] static Result<ReferenceResult> run(
      const VerifiedGraph& graph, std::span<const InputBinding> bindings,
      ReferenceLimits limits = ReferenceLimits{});
};

}  // namespace tensorkiln
