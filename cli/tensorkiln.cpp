#include <cstddef>
#include <exception>
#include <iostream>
#include <span>
#include <string_view>
#include <vector>

#include "cli_app.hpp"

int main(const int argc, const char* const argv[]) {
  try {
    std::vector<std::string_view> arguments;
    if (argc > 1) {
      arguments.reserve(static_cast<std::size_t>(argc - 1));
    }
    for (int index = 1; index < argc; ++index) {
      arguments.emplace_back(argv[index]);
    }
    return tensorkiln::cli::run(arguments, std::cout, std::cerr);
  } catch (const std::exception& exception) {
    std::cerr << "tensorkiln: internal failure: " << exception.what() << '\n';
    return tensorkiln::cli::kExitInternalFailure;
  } catch (...) {
    std::cerr << "tensorkiln: internal failure: non-standard exception\n";
    return tensorkiln::cli::kExitInternalFailure;
  }
}
