#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <new>
#include <optional>
#include <span>
#include <string_view>
#include <utility>

#include "tensorkiln/execution.hpp"

namespace {

thread_local bool allocation_audit_armed = false;
thread_local std::uint64_t audited_allocations = 0U;

void count_audited_allocation() noexcept {
  if (allocation_audit_armed) {
    ++audited_allocations;
  }
}

[[nodiscard]] void* allocate_bytes(std::size_t size) {
  if (size == 0U) {
    size = 1U;
  }
  if (void* const result = std::malloc(size); result != nullptr) {
    return result;
  }
  throw std::bad_alloc{};
}

[[nodiscard]] void* allocate_aligned_bytes(std::size_t size,
                                           const std::size_t alignment) {
  if (size == 0U) {
    size = 1U;
  }
  if (size > std::numeric_limits<std::size_t>::max() - (alignment - 1U)) {
    throw std::bad_alloc{};
  }
  const std::size_t rounded =
      ((size + alignment - 1U) / alignment) * alignment;
  if (void* const result = std::aligned_alloc(alignment, rounded);
      result != nullptr) {
    return result;
  }
  throw std::bad_alloc{};
}

template <typename T>
[[nodiscard]] T unwrap(tensorkiln::Result<T> result) {
  if (result.value_if() == nullptr) {
    std::abort();
  }
  return std::move(*result.value_if());
}

[[nodiscard]] tensorkiln::TensorType f32(
    const std::initializer_list<std::int64_t> extents) {
  return unwrap(tensorkiln::TensorType::create(
      unwrap(tensorkiln::Shape::create(extents))));
}

struct ExecutionFixture final {
  tensorkiln::VerifiedGraph graph;
  std::array<float, 4U> left{{1.0F, 2.0F, -3.0F, 4.0F}};
  std::array<float, 4U> right{{0.5F, -1.0F, 2.0F, 3.0F}};
  std::array<float, 8U> batched{{
      1.0F, 0.0F, 0.0F, 1.0F,
      -1.0F, 2.0F, 3.0F, -4.0F,
  }};
};

[[nodiscard]] ExecutionFixture make_execution_fixture() {
  tensorkiln::GraphBuilder builder;
  const tensorkiln::ValueId left =
      unwrap(builder.input("left", f32({2, 2})));
  const tensorkiln::ValueId right =
      unwrap(builder.input("right", f32({2, 2})));
  const tensorkiln::ValueId contiguous = unwrap(builder.add(left, right));

  const std::array<float, 2U> bias_data{{0.25F, -0.5F}};
  const tensorkiln::ValueId bias =
      unwrap(builder.constant("bias", f32({2}), bias_data));
  const tensorkiln::ValueId broadcast =
      unwrap(builder.add(contiguous, bias));

  const std::array<float, 4U> weight_data{{1.0F, -2.0F, 0.5F, 3.0F}};
  const tensorkiln::ValueId weight =
      unwrap(builder.constant("weight", f32({2, 2}), weight_data));
  const tensorkiln::ValueId rank2 =
      unwrap(builder.matmul(broadcast, weight));
  const tensorkiln::ValueId rank2_result = unwrap(builder.relu(rank2));

  const tensorkiln::ValueId batched =
      unwrap(builder.input("batched", f32({2, 1, 2, 2})));
  const std::array<float, 12U> batched_weight_data{{
      1.0F, 0.0F, 0.0F, 1.0F,
      0.5F, -1.0F, 2.0F, 0.25F,
      -2.0F, 1.0F, 1.5F, -0.5F,
  }};
  const tensorkiln::ValueId batched_weight = unwrap(builder.constant(
      "batched_weight", f32({1, 3, 2, 2}), batched_weight_data));
  const tensorkiln::ValueId batched_product =
      unwrap(builder.matmul(batched, batched_weight));
  const tensorkiln::ValueId batched_result =
      unwrap(builder.relu(batched_product));

  static_cast<void>(unwrap(builder.output("rank2", rank2_result)));
  static_cast<void>(unwrap(builder.output("batched", batched_result)));
  return ExecutionFixture{unwrap(std::move(builder).finish())};
}

[[nodiscard]] bool mix_output(
    const tensorkiln::ExecutionSession& session,
    const std::string_view name, std::uint64_t& checksum) noexcept {
  const std::optional<tensorkiln::ExecutionResultView> result =
      session.result();
  if (!result.has_value()) {
    return false;
  }
  const std::optional<tensorkiln::TensorView> output = result->output(name);
  if (!output.has_value()) {
    return false;
  }
  for (const float value : output->data()) {
    checksum ^= static_cast<std::uint64_t>(std::bit_cast<std::uint32_t>(value));
    checksum *= UINT64_C(1099511628211);
  }
  return true;
}

[[nodiscard]] bool run_and_observe(
    tensorkiln::ExecutionSession& session,
    const bool external_only, std::uint64_t& checksum) noexcept {
  if (session.run() != tensorkiln::ExecutionRunStatus::success) {
    return false;
  }
  if (external_only) {
    return mix_output(session, "external", checksum);
  }
  return mix_output(session, "rank2", checksum) &&
         mix_output(session, "batched", checksum);
}

[[nodiscard]] int audit_execution() {
  const ExecutionFixture fixture = make_execution_fixture();
  const tensorkiln::ExecutionPlan plan =
      unwrap(tensorkiln::ExecutionPlanCompiler::run(fixture.graph));
  std::array<std::uint64_t, 5U> kernel_counts{};
  for (const tensorkiln::ExecutionStep& step : plan.steps()) {
    const std::size_t kernel = static_cast<std::size_t>(step.kernel());
    if (kernel >= kernel_counts.size()) {
      return 10;
    }
    ++kernel_counts[kernel];
  }
  for (const std::uint64_t count : kernel_counts) {
    if (count == 0U) {
      return 11;
    }
  }

  const std::array<tensorkiln::ExecutionInputBinding, 3U> bindings{{
      {"left", fixture.left},
      {"right", fixture.right},
      {"batched", fixture.batched},
  }};
  tensorkiln::ExecutionSession regular =
      tensorkiln::ExecutionSession::create(plan);
  tensorkiln::ExecutionSession audited =
      tensorkiln::ExecutionSession::create(
          plan, tensorkiln::ExecutionSessionOptions{true});
  if (!regular.bind(bindings).has_value() ||
      !audited.bind(bindings).has_value() ||
      !audited.audits_kernel_writes()) {
    return 12;
  }

  tensorkiln::GraphBuilder external_builder;
  const tensorkiln::ValueId external_value =
      unwrap(external_builder.input("x", f32({3})));
  static_cast<void>(
      unwrap(external_builder.output("external", external_value)));
  const tensorkiln::VerifiedGraph external_graph =
      unwrap(std::move(external_builder).finish());
  const tensorkiln::ExecutionPlan external_plan =
      unwrap(tensorkiln::ExecutionPlanCompiler::run(external_graph));
  const std::array<float, 3U> external_data{{1.0F, -2.0F, 3.0F}};
  const std::array<tensorkiln::ExecutionInputBinding, 1U> external_bindings{{
      {"x", external_data},
  }};
  tensorkiln::ExecutionSession external =
      tensorkiln::ExecutionSession::create(
          external_plan, tensorkiln::ExecutionSessionOptions{true});
  if (!external.bind(external_bindings).has_value() ||
      external.workspace_bytes() != 0U) {
    return 13;
  }

  audited_allocations = 0U;
  allocation_audit_armed = true;
  std::uint64_t first_checksum = UINT64_C(1469598103934665603);
  const bool first_ok =
      run_and_observe(regular, false, first_checksum) &&
      run_and_observe(audited, false, first_checksum) &&
      run_and_observe(external, true, first_checksum);
  std::uint64_t second_checksum = UINT64_C(1469598103934665603);
  const bool second_ok =
      run_and_observe(regular, false, second_checksum) &&
      run_and_observe(audited, false, second_checksum) &&
      run_and_observe(external, true, second_checksum);
  allocation_audit_armed = false;

  if (!first_ok || !second_ok || first_checksum != second_checksum) {
    return 14;
  }
  if (audited_allocations != 0U) {
    std::fprintf(stderr,
                 "execution allocation audit observed %llu allocations\n",
                 static_cast<unsigned long long>(audited_allocations));
    return 15;
  }
  return 0;
}

}  // namespace

extern "C" {

void* __real_malloc(std::size_t size);
void* __real_calloc(std::size_t count, std::size_t size);
void* __real_realloc(void* pointer, std::size_t size);
void* __real_aligned_alloc(std::size_t alignment, std::size_t size);
int __real_posix_memalign(void** pointer, std::size_t alignment,
                          std::size_t size);

void* __wrap_malloc(const std::size_t size) {
  count_audited_allocation();
  return __real_malloc(size);
}

void* __wrap_calloc(const std::size_t count, const std::size_t size) {
  count_audited_allocation();
  return __real_calloc(count, size);
}

void* __wrap_realloc(void* const pointer, const std::size_t size) {
  count_audited_allocation();
  return __real_realloc(pointer, size);
}

void* __wrap_aligned_alloc(const std::size_t alignment,
                           const std::size_t size) {
  count_audited_allocation();
  return __real_aligned_alloc(alignment, size);
}

int __wrap_posix_memalign(void** const pointer, const std::size_t alignment,
                          const std::size_t size) {
  count_audited_allocation();
  return __real_posix_memalign(pointer, alignment, size);
}

}  // extern "C"

void* operator new(const std::size_t size) { return allocate_bytes(size); }

void* operator new[](const std::size_t size) { return allocate_bytes(size); }

void operator delete(void* const pointer) noexcept { std::free(pointer); }

void operator delete[](void* const pointer) noexcept { std::free(pointer); }

void operator delete(void* const pointer, std::size_t) noexcept {
  std::free(pointer);
}

void operator delete[](void* const pointer, std::size_t) noexcept {
  std::free(pointer);
}

void* operator new(const std::size_t size, const std::align_val_t alignment) {
  return allocate_aligned_bytes(size, static_cast<std::size_t>(alignment));
}

void* operator new[](const std::size_t size, const std::align_val_t alignment) {
  return allocate_aligned_bytes(size, static_cast<std::size_t>(alignment));
}

void operator delete(void* const pointer, std::align_val_t) noexcept {
  std::free(pointer);
}

void operator delete[](void* const pointer, std::align_val_t) noexcept {
  std::free(pointer);
}

void operator delete(void* const pointer, std::size_t,
                     std::align_val_t) noexcept {
  std::free(pointer);
}

void operator delete[](void* const pointer, std::size_t,
                       std::align_val_t) noexcept {
  std::free(pointer);
}

void* operator new(const std::size_t size,
                   const std::nothrow_t&) noexcept {
  try {
    return ::operator new(size);
  } catch (...) {
    return nullptr;
  }
}

void* operator new[](const std::size_t size,
                     const std::nothrow_t&) noexcept {
  try {
    return ::operator new[](size);
  } catch (...) {
    return nullptr;
  }
}

void operator delete(void* const pointer,
                     const std::nothrow_t&) noexcept {
  ::operator delete(pointer);
}

void operator delete[](void* const pointer,
                       const std::nothrow_t&) noexcept {
  ::operator delete[](pointer);
}

void* operator new(const std::size_t size, const std::align_val_t alignment,
                   const std::nothrow_t&) noexcept {
  try {
    return ::operator new(size, alignment);
  } catch (...) {
    return nullptr;
  }
}

void* operator new[](const std::size_t size,
                     const std::align_val_t alignment,
                     const std::nothrow_t&) noexcept {
  try {
    return ::operator new[](size, alignment);
  } catch (...) {
    return nullptr;
  }
}

void operator delete(void* const pointer, std::align_val_t alignment,
                     const std::nothrow_t&) noexcept {
  ::operator delete(pointer, alignment);
}

void operator delete[](void* const pointer, std::align_val_t alignment,
                       const std::nothrow_t&) noexcept {
  ::operator delete[](pointer, alignment);
}

int main() { return audit_execution(); }
