#include "test.hpp"

#include <cstdlib>
#include <exception>
#include <iostream>

int main() {
  std::size_t failures = 0U;

  for (const auto& test_case : tensorkiln::test::registry()) {
    try {
      test_case.function();
      std::cout << "[pass] " << test_case.name << '\n';
    } catch (const std::exception& error) {
      ++failures;
      std::cerr << "[fail] " << test_case.name << "\n  " << error.what()
                << '\n';
    } catch (...) {
      ++failures;
      std::cerr << "[fail] " << test_case.name
                << "\n  unknown non-standard exception\n";
    }
  }

  const std::size_t total = tensorkiln::test::registry().size();
  std::cout << "\n" << (total - failures) << '/' << total << " tests passed\n";
  return failures == 0U ? EXIT_SUCCESS : EXIT_FAILURE;
}
