#pragma once

#include <source_location>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace tensorkiln::test {

using TestFunction = void (*)();

struct TestCase final {
  std::string_view name;
  TestFunction function;
};

[[nodiscard]] inline std::vector<TestCase>& registry() {
  static std::vector<TestCase> cases;
  return cases;
}

class Registrar final {
 public:
  Registrar(const std::string_view name, const TestFunction function) {
    registry().push_back(TestCase{name, function});
  }
};

class Failure final : public std::runtime_error {
 public:
  explicit Failure(std::string message) : std::runtime_error(std::move(message)) {}
};

inline void require(
    const bool condition, const std::string_view expression,
    const std::source_location location = std::source_location::current()) {
  if (!condition) {
    throw Failure(std::string(location.file_name()) + ":" +
                  std::to_string(location.line()) + ": requirement failed: " +
                  std::string(expression));
  }
}

template <typename Left, typename Right>
void require_equal(
    const Left& left, const Right& right, const std::string_view expression,
    const std::source_location location = std::source_location::current()) {
  require(left == right, expression, location);
}

}  // namespace tensorkiln::test

#define TK_DETAIL_JOIN_INNER(left, right) left##right
#define TK_DETAIL_JOIN(left, right) TK_DETAIL_JOIN_INNER(left, right)

#define TK_TEST(name)                                                        \
  static void TK_DETAIL_JOIN(tk_test_function_, __LINE__)();                 \
  [[maybe_unused]] static const ::tensorkiln::test::Registrar                \
      TK_DETAIL_JOIN(tk_test_registrar_, __LINE__)(                          \
          name, &TK_DETAIL_JOIN(tk_test_function_, __LINE__));               \
  static void TK_DETAIL_JOIN(tk_test_function_, __LINE__)()

#define TK_REQUIRE(expression)                                               \
  ::tensorkiln::test::require(static_cast<bool>(expression), #expression)

#define TK_REQUIRE_EQ(left, right)                                           \
  ::tensorkiln::test::require_equal((left), (right), #left " == " #right)
