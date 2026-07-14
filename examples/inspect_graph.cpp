#include <array>
#include <cstddef>
#include <cstdlib>
#include <cstdint>
#include <exception>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "tensorkiln/reference.hpp"

namespace {

template <typename T>
[[nodiscard]] T unwrap(tensorkiln::Result<T> result) {
  if (result.value_if() == nullptr) {
    const tensorkiln::Diagnostic& error = *result.error_if();
    throw std::runtime_error(
        std::string(tensorkiln::error_code_name(error.code)) + ": " +
        error.message);
  }
  return std::move(*result.value_if());
}

[[nodiscard]] tensorkiln::TensorType f32(
    const std::initializer_list<std::int64_t> extents) {
  tensorkiln::Shape shape = unwrap(tensorkiln::Shape::create(extents));
  return unwrap(tensorkiln::TensorType::create(std::move(shape)));
}

}  // namespace

int main() {
  try {
    tensorkiln::GraphBuilder builder;
    const tensorkiln::ValueId x = unwrap(builder.input("x", f32({2, 3})));
    const std::vector<float> bias_data{1.0F, -0.0F, 2.0F};
    const tensorkiln::ValueId bias =
        unwrap(builder.constant("bias", f32({3}), bias_data));
    const tensorkiln::ValueId sum = unwrap(builder.add(x, bias));
    const tensorkiln::ValueId result = unwrap(builder.relu(sum));
    static_cast<void>(unwrap(builder.output("result", result)));

    const tensorkiln::VerifiedGraph graph =
        unwrap(std::move(builder).finish());
    std::cout << graph.dump();

    const std::array<float, 6U> input_data{{
        -2.0F, 1.0F, 3.0F, -1.0F, 0.5F, -4.0F,
    }};
    const std::array<tensorkiln::InputBinding, 1U> bindings{{
        tensorkiln::InputBinding{"x", input_data},
    }};
    const tensorkiln::ReferenceResult execution =
        unwrap(tensorkiln::ReferenceInterpreter::run(graph, bindings));
    const tensorkiln::Tensor* output = execution.output("result");
    if (output == nullptr) {
      throw std::runtime_error("reference result has no @result output");
    }
    std::cout << "result = [";
    for (std::size_t index = 0U; index < output->data().size(); ++index) {
      std::cout << (index == 0U ? "" : ", ") << output->data()[index];
    }
    std::cout << "]\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "TensorKiln example failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
