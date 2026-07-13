#pragma once

#include <type_traits>
#include <utility>
#include <variant>

#include "tensorkiln/diagnostic.hpp"

namespace tensorkiln {

template <typename T>
class [[nodiscard]] Result final {
  static_assert(std::is_object_v<T>);
  static_assert(!std::is_const_v<T>);
  static_assert(!std::is_same_v<T, Diagnostic>);

 public:
  [[nodiscard]] static Result success(T value) {
    return Result(std::in_place_index<0>, std::move(value));
  }

  [[nodiscard]] static Result failure(Diagnostic diagnostic) {
    return Result(std::in_place_index<1>, std::move(diagnostic));
  }

  Result(const Result&) = default;
  Result(Result&&) = default;
  Result& operator=(const Result&) = delete;
  Result& operator=(Result&&) = delete;
  ~Result() = default;

  [[nodiscard]] bool has_value() const noexcept {
    return std::holds_alternative<T>(storage_);
  }

  [[nodiscard]] explicit operator bool() const noexcept {
    return has_value();
  }

  [[nodiscard]] T* value_if() noexcept { return std::get_if<T>(&storage_); }

  [[nodiscard]] const T* value_if() const noexcept {
    return std::get_if<T>(&storage_);
  }

  [[nodiscard]] Diagnostic* error_if() noexcept {
    return std::get_if<Diagnostic>(&storage_);
  }

  [[nodiscard]] const Diagnostic* error_if() const noexcept {
    return std::get_if<Diagnostic>(&storage_);
  }

 private:
  explicit Result(std::in_place_index_t<0>, T value)
      : storage_(std::in_place_index<0>, std::move(value)) {}

  explicit Result(std::in_place_index_t<1>, Diagnostic diagnostic)
      : storage_(std::in_place_index<1>, std::move(diagnostic)) {}

  std::variant<T, Diagnostic> storage_;
};

}  // namespace tensorkiln
