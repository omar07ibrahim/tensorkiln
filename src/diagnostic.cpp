#include "tensorkiln/diagnostic.hpp"

namespace tensorkiln {

std::string_view error_code_name(const ErrorCode code) noexcept {
  switch (code) {
    case ErrorCode::rank_limit_exceeded:
      return "rank_limit_exceeded";
    case ErrorCode::extent_not_positive:
      return "extent_not_positive";
    case ErrorCode::element_count_overflow:
      return "element_count_overflow";
    case ErrorCode::element_limit_exceeded:
      return "element_limit_exceeded";
    case ErrorCode::broadcast_incompatible:
      return "broadcast_incompatible";
    case ErrorCode::matmul_rank_unsupported:
      return "matmul_rank_unsupported";
    case ErrorCode::matmul_inner_dimension_mismatch:
      return "matmul_inner_dimension_mismatch";
    case ErrorCode::invalid_name:
      return "invalid_name";
    case ErrorCode::name_limit_exceeded:
      return "name_limit_exceeded";
    case ErrorCode::duplicate_name:
      return "duplicate_name";
    case ErrorCode::value_not_found:
      return "value_not_found";
    case ErrorCode::constant_size_mismatch:
      return "constant_size_mismatch";
    case ErrorCode::graph_node_limit_exceeded:
      return "graph_node_limit_exceeded";
    case ErrorCode::graph_output_limit_exceeded:
      return "graph_output_limit_exceeded";
    case ErrorCode::constant_element_limit_exceeded:
      return "constant_element_limit_exceeded";
    case ErrorCode::graph_has_no_outputs:
      return "graph_has_no_outputs";
    case ErrorCode::builder_finished:
      return "builder_finished";
    case ErrorCode::unsupported_element_type:
      return "unsupported_element_type";
    case ErrorCode::byte_count_overflow:
      return "byte_count_overflow";
    case ErrorCode::byte_limit_exceeded:
      return "byte_limit_exceeded";
    case ErrorCode::input_binding_count_exceeded:
      return "input_binding_count_exceeded";
    case ErrorCode::input_binding_unknown:
      return "input_binding_unknown";
    case ErrorCode::input_binding_duplicate:
      return "input_binding_duplicate";
    case ErrorCode::input_binding_missing:
      return "input_binding_missing";
    case ErrorCode::input_binding_size_mismatch:
      return "input_binding_size_mismatch";
    case ErrorCode::unsupported_rounding_mode:
      return "unsupported_rounding_mode";
    case ErrorCode::unsupported_subnormal_mode:
      return "unsupported_subnormal_mode";
    case ErrorCode::reference_materialization_limit_exceeded:
      return "reference_materialization_limit_exceeded";
    case ErrorCode::reference_scalar_step_limit_exceeded:
      return "reference_scalar_step_limit_exceeded";
  }
  return "unknown_error";
}

}  // namespace tensorkiln
