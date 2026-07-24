#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <utility>

#include "tensorkiln/execution_plan.hpp"

namespace tensorkiln {

struct ExecutionInputBinding final {
  std::string_view name;
  std::span<const float> data;
};

struct ExecutionSessionOptions final {
  // When enabled, run() snapshots the arena before every kernel and verifies
  // that only the exact result payload changed. The shadow is allocated by
  // create(), so audited execution remains heap-allocation-free.
  bool audit_kernel_writes = false;
};

class BoundInputs final {
 public:
  [[nodiscard]] std::uint32_t input_count() const noexcept {
    return input_count_;
  }

 private:
  friend class ExecutionSession;

  explicit BoundInputs(std::uint32_t input_count) noexcept
      : input_count_(input_count) {}

  std::uint32_t input_count_;
};

enum class ExecutionRunStatus : std::uint8_t {
  success,
  inputs_not_bound,
  unsupported_rounding_mode,
  unsupported_binary64_precision,
  unsupported_subnormal_mode,
  memory_corruption,
};

[[nodiscard]] std::string_view execution_run_status_name(
    ExecutionRunStatus status) noexcept;

class TensorView final {
 public:
  // TensorView is a non-owning snapshot. type() and data() may be used only
  // while the originating ExecutionResultView is current. A span copied from
  // data() has the same lifetime restriction.
  [[nodiscard]] const TensorType& type() const noexcept { return *type_; }
  [[nodiscard]] std::span<const float> data() const noexcept { return data_; }

 private:
  friend class ExecutionSession;

  TensorView(const TensorType& type, std::span<const float> data) noexcept
      : type_(&type), data_(data) {}

  const TensorType* type_;
  std::span<const float> data_;
};

class ExecutionSession;

namespace detail {
struct ExecutionResultState;
class ExecutionSessionData;
enum class ExecutionFaultKind : std::uint8_t;
void set_execution_fault_for_test(ExecutionSession& session,
                                  ExecutionFaultKind fault) noexcept;
}

// A result view borrows its session. Binding new inputs, starting another run,
// moving the session, or destroying the session makes the view stale. Output
// lookup on a stale view returns std::nullopt.
class ExecutionResultView final {
 public:
  [[nodiscard]] bool current() const noexcept;
  [[nodiscard]] std::optional<TensorView> output(
      std::string_view name) const noexcept;

 private:
  friend class ExecutionSession;

  ExecutionResultView(
      std::weak_ptr<const detail::ExecutionResultState> state,
      std::uint64_t generation) noexcept
      : state_(std::move(state)), generation_(generation) {}

  std::weak_ptr<const detail::ExecutionResultState> state_;
  std::uint64_t generation_;
};

// A session borrows one immutable ExecutionPlan, which must not be moved or
// destroyed while the session exists. Session construction allocates all
// workspace and pointer metadata. bind() performs validation outside the hot
// path; a successful run() is synchronous, noexcept, and heap-allocation-free.
// One session is deliberately not thread-safe, while independent sessions may
// execute the same plan concurrently. A moved-from session supports only
// destruction; all observers and operations require a non-moved-from object.
class ExecutionSession final {
 public:
  [[nodiscard]] static ExecutionSession create(
      const ExecutionPlan& plan,
      ExecutionSessionOptions options = ExecutionSessionOptions{});

  ExecutionSession(const ExecutionSession&) = delete;
  ExecutionSession(ExecutionSession&& other) noexcept;
  ExecutionSession& operator=(const ExecutionSession&) = delete;
  ExecutionSession& operator=(ExecutionSession&&) = delete;
  ~ExecutionSession();

  [[nodiscard]] const ExecutionPlan& plan() const noexcept;
  [[nodiscard]] std::uint64_t workspace_bytes() const noexcept;
  [[nodiscard]] bool audits_kernel_writes() const noexcept;

  // Binding data is borrowed, and a successful binding remains active until
  // the next bind() or session destruction. Every payload must remain alive
  // and unchanged throughout that interval. This also keeps Input-backed
  // outputs valid while their corresponding result view is current.
  [[nodiscard]] Result<BoundInputs> bind(
      std::span<const ExecutionInputBinding> bindings);

  [[nodiscard]] ExecutionRunStatus run() noexcept;
  [[nodiscard]] std::optional<ExecutionResultView> result() const noexcept;

 private:
  friend class ExecutionResultView;
  friend void detail::set_execution_fault_for_test(
      ExecutionSession& session, detail::ExecutionFaultKind fault) noexcept;

  explicit ExecutionSession(
      std::unique_ptr<detail::ExecutionSessionData> data) noexcept;

  [[nodiscard]] bool result_is_current(
      std::uint64_t generation) const noexcept;
  [[nodiscard]] std::optional<TensorView> output(
      std::string_view name, std::uint64_t generation) const noexcept;

  std::unique_ptr<detail::ExecutionSessionData> data_;
};

}  // namespace tensorkiln
