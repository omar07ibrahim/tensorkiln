#include <array>
#include <bit>
#include <cstddef>
#include <cstdlib>
#include <cstdint>
#include <exception>
#include <initializer_list>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "tensorkiln/dead_code_elimination.hpp"
#include "tensorkiln/reference.hpp"
#include "tensorkiln/structural_canonicalization.hpp"

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

[[nodiscard]] bool same_bits(const std::span<const float> left,
                             const std::span<const float> right) {
  if (left.size() != right.size()) {
    return false;
  }
  for (std::size_t index = 0U; index < left.size(); ++index) {
    if (std::bit_cast<std::uint32_t>(left[index]) !=
        std::bit_cast<std::uint32_t>(right[index])) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] const tensorkiln::Tensor& require_output(
    const tensorkiln::ReferenceResult& execution,
    const std::string_view name) {
  const tensorkiln::Tensor* output = execution.output(name);
  if (output == nullptr) {
    throw std::runtime_error(
        "reference result has no @" + std::string(name) + " output");
  }
  return *output;
}

void print_tensor(const std::string_view name,
                  const tensorkiln::Tensor& tensor) {
  std::cout << name << " = [";
  for (std::size_t index = 0U; index < tensor.data().size(); ++index) {
    std::cout << (index == 0U ? "" : ", ") << tensor.data()[index];
  }
  std::cout << "]\n";
}

}  // namespace

int main() {
  try {
    tensorkiln::GraphBuilder builder;
    const tensorkiln::ValueId x = unwrap(builder.input("x", f32({2, 3})));
    const std::vector<float> bias_data{1.0F, -0.0F, 2.0F};
    const tensorkiln::ValueId bias =
        unwrap(builder.constant("bias", f32({3}), bias_data));
    const std::vector<float> dead_data{10.0F, 20.0F, 30.0F};
    const tensorkiln::ValueId dead_bias =
        unwrap(builder.constant("dead_bias", f32({3}), dead_data));
    const tensorkiln::ValueId dead_sum = unwrap(builder.add(x, dead_bias));
    const tensorkiln::ValueId dead_relu = unwrap(builder.relu(dead_sum));

    const tensorkiln::ValueId first_sum = unwrap(builder.add(x, bias));
    const tensorkiln::ValueId second_sum = unwrap(builder.add(x, bias));
    const tensorkiln::ValueId first_relu = unwrap(builder.relu(first_sum));
    const tensorkiln::ValueId second_relu = unwrap(builder.relu(second_sum));
    const tensorkiln::ValueId nested_relu =
        unwrap(builder.relu(second_relu));
    const tensorkiln::ValueId result =
        unwrap(builder.add(first_relu, nested_relu));
    static_cast<void>(unwrap(builder.output("result", result)));
    static_cast<void>(unwrap(builder.output("result_alias", result)));

    const tensorkiln::VerifiedGraph source =
        unwrap(std::move(builder).finish());
    std::cout << "=== source graph ===\n" << source.dump();

    const tensorkiln::DeadCodeEliminationResult eliminated =
        unwrap(tensorkiln::DeadCodeElimination::run(source));
    std::cout << "=== dead-code elimination ===\n" << eliminated.dump();

    const tensorkiln::StructuralCanonicalizationResult canonicalized =
        unwrap(tensorkiln::StructuralCanonicalization::run(
            eliminated.graph(), eliminated.provenance()));
    const tensorkiln::NodeProvenance* first_sum_lineage =
        canonicalized.provenance().for_source(first_sum);
    const tensorkiln::NodeProvenance* second_sum_lineage =
        canonicalized.provenance().for_source(second_sum);
    const tensorkiln::NodeProvenance* first_relu_lineage =
        canonicalized.provenance().for_source(first_relu);
    const tensorkiln::NodeProvenance* second_relu_lineage =
        canonicalized.provenance().for_source(second_relu);
    const tensorkiln::NodeProvenance* nested_relu_lineage =
        canonicalized.provenance().for_source(nested_relu);
    if (first_sum_lineage == nullptr ||
        first_sum_lineage != second_sum_lineage ||
        first_relu_lineage == nullptr ||
        first_relu_lineage != second_relu_lineage ||
        first_relu_lineage != nested_relu_lineage) {
      throw std::runtime_error(
          "composed provenance lost a canonicalized source definition");
    }
    if (canonicalized.provenance().for_source(dead_bias) != nullptr ||
        canonicalized.provenance().for_source(dead_sum) != nullptr ||
        canonicalized.provenance().for_source(dead_relu) != nullptr) {
      throw std::runtime_error(
          "composed provenance retained a dead source definition");
    }
    std::cout << "=== structural canonicalization ===\n"
              << canonicalized.dump();
    std::cout << "=== rewritten graph ===\n" << canonicalized.graph().dump();

    const std::array<float, 6U> input_data{{
        -2.0F, 1.0F, 3.0F, -1.0F, 0.5F, -4.0F,
    }};
    const std::array<tensorkiln::InputBinding, 1U> bindings{{
        tensorkiln::InputBinding{"x", input_data},
    }};
    const tensorkiln::ReferenceResult source_execution =
        unwrap(tensorkiln::ReferenceInterpreter::run(source, bindings));
    const tensorkiln::ReferenceResult rewritten_execution =
        unwrap(tensorkiln::ReferenceInterpreter::run(
            canonicalized.graph(), bindings));
    const tensorkiln::Tensor& source_output =
        require_output(source_execution, "result");
    const tensorkiln::Tensor& rewritten_output =
        require_output(rewritten_execution, "result");
    if (!same_bits(source_output.data(), rewritten_output.data())) {
      throw std::runtime_error(
          "source and rewritten outputs differ at the bit level");
    }
    if (source_execution.output("result") !=
            source_execution.output("result_alias") ||
        rewritten_execution.output("result") !=
            rewritten_execution.output("result_alias")) {
      throw std::runtime_error("compiler changed output alias topology");
    }
    print_tensor("result", rewritten_output);
    std::cout << "verified: bit-exact output and preserved output alias\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "TensorKiln example failed: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
