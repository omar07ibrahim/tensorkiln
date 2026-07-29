#include "cli_app.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <initializer_list>
#include <ios>
#include <limits>
#include <locale>
#include <ostream>
#include <sstream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "tensorkiln/execution_plan.hpp"

namespace tensorkiln::cli {
namespace {

constexpr std::string_view kInspectSchema = "tensorkiln.cli.inspect.v1";
constexpr std::string_view kListSchema = "tensorkiln.cli.workloads.v1";
constexpr std::string_view kErrorSchema = "tensorkiln.cli.error.v1";
constexpr std::string_view kWorkloadId = "dense_relu_v1";
constexpr std::string_view kWorkloadDescription =
    "f32[2,3] -> MatMul(f32[3,2]) -> Add(f32[2]) -> Relu";
constexpr std::size_t kMaxArgumentCount = 32U;
constexpr std::size_t kMaxArgumentBytes = 8192U;
constexpr std::size_t kMaxTotalArgumentBytes = 16384U;

enum class OutputFormat : std::uint8_t {
  text,
  json,
};

class CliFailure final : public std::runtime_error {
 public:
  CliFailure(const int exit_code, std::string code, std::string message)
      : std::runtime_error(std::move(message)),
        exit_code_(exit_code),
        code_(std::move(code)) {}

  [[nodiscard]] int exit_code() const noexcept { return exit_code_; }
  [[nodiscard]] const std::string& code() const noexcept { return code_; }

 private:
  int exit_code_;
  std::string code_;
};

[[noreturn]] void usage_failure(std::string code, std::string message) {
  throw CliFailure(kExitUsage, std::move(code), std::move(message));
}

[[noreturn]] void build_failure(const std::string_view stage,
                                const Diagnostic& diagnostic) {
  throw CliFailure(
      kExitBuildFailure, "workload_build_failed",
      std::string(stage) + " failed with " +
          std::string(error_code_name(diagnostic.code)) + ": " +
          diagnostic.message);
}

template <typename T>
[[nodiscard]] T unwrap(Result<T> result, const std::string_view stage) {
  if (result.value_if() == nullptr) {
    build_failure(stage, *result.error_if());
  }
  return std::move(*result.value_if());
}

[[nodiscard]] TensorType f32(
    const std::initializer_list<std::int64_t> extents) {
  Shape shape = unwrap(Shape::create(extents), "shape construction");
  return unwrap(TensorType::create(std::move(shape)),
                "tensor type construction");
}

[[nodiscard]] VerifiedGraph build_dense_relu_graph() {
  GraphBuilder builder;
  const ValueId input =
      unwrap(builder.input("x", f32({2, 3})), "input construction");
  constexpr std::array<float, 6U> weights{{
      1.0F,
      -2.0F,
      3.0F,
      4.0F,
      -1.0F,
      2.0F,
  }};
  const ValueId weight = unwrap(
      builder.constant("weight", f32({3, 2}), weights),
      "weight construction");
  const ValueId product =
      unwrap(builder.matmul(input, weight), "MatMul construction");
  constexpr std::array<float, 2U> bias_values{{0.5F, -1.0F}};
  const ValueId bias = unwrap(
      builder.constant("bias", f32({2}), bias_values),
      "bias construction");
  const ValueId shifted =
      unwrap(builder.add(product, bias), "Add construction");
  const ValueId result =
      unwrap(builder.relu(shifted), "Relu construction");
  static_cast<void>(
      unwrap(builder.output("result", result), "output construction"));
  return unwrap(std::move(builder).finish(), "graph finalization");
}

[[nodiscard]] ExecutionPlan build_dense_relu_plan() {
  const VerifiedGraph graph = build_dense_relu_graph();
  return unwrap(ExecutionPlanCompiler::run(graph), "plan compilation");
}

void write_json_string(std::ostream& output, const std::string_view value) {
  constexpr char kHex[] = "0123456789abcdef";
  output.put('"');
  for (const char raw_character : value) {
    const auto character = static_cast<unsigned char>(raw_character);
    switch (character) {
      case '"':
        output << "\\\"";
        break;
      case '\\':
        output << "\\\\";
        break;
      case '\b':
        output << "\\b";
        break;
      case '\f':
        output << "\\f";
        break;
      case '\n':
        output << "\\n";
        break;
      case '\r':
        output << "\\r";
        break;
      case '\t':
        output << "\\t";
        break;
      default:
        if (character < 0x20U) {
          output << "\\u00" << kHex[(character >> 4U) & 0x0fU]
                 << kHex[character & 0x0fU];
        } else {
          output.put(static_cast<char>(character));
        }
        break;
    }
  }
  output.put('"');
}

[[nodiscard]] bool requests_json_errors(
    const std::span<const std::string_view> arguments) noexcept {
  for (std::size_t index = 0U; index < arguments.size(); ++index) {
    if (arguments[index] == "--format=json") {
      return true;
    }
    if (arguments[index] == "--format" && index + 1U < arguments.size() &&
        arguments[index + 1U] == "json") {
      return true;
    }
  }
  return false;
}

void validate_argument_envelope(
    const std::span<const std::string_view> arguments) {
  if (arguments.size() > kMaxArgumentCount) {
    usage_failure("argument_limit_exceeded",
                  "the command has more than 32 arguments");
  }
  std::size_t total_bytes = 0U;
  for (const std::string_view argument : arguments) {
    if (argument.size() > kMaxArgumentBytes ||
        total_bytes > kMaxTotalArgumentBytes - argument.size()) {
      usage_failure("argument_limit_exceeded",
                    "command arguments exceed the 16384-byte limit");
    }
    total_bytes += argument.size();
    for (const char raw_character : argument) {
      const auto character = static_cast<unsigned char>(raw_character);
      if (character < 0x20U || character > 0x7eU) {
        usage_failure("invalid_argument_encoding",
                      "command arguments must contain printable ASCII only");
      }
    }
  }
}

void write_error(std::ostream& error, const CliFailure& failure,
                 const bool json) {
  if (!json) {
    error << "tensorkiln: " << failure.code() << ": " << failure.what()
          << '\n';
    return;
  }
  error << "{\"schema\":";
  write_json_string(error, kErrorSchema);
  error << ",\"error\":{\"code\":";
  write_json_string(error, failure.code());
  error << ",\"message\":";
  write_json_string(error, failure.what());
  error << "}}\n";
}

[[nodiscard]] std::ostringstream make_canonical_stream() {
  std::ostringstream stream;
  stream.imbue(std::locale::classic());
  return stream;
}

[[nodiscard]] bool publish(std::ostream& destination,
                           const std::string& payload) noexcept {
  if (payload.size() >
      static_cast<std::size_t>(
          std::numeric_limits<std::streamsize>::max())) {
    return false;
  }
  try {
    destination.write(
        payload.data(), static_cast<std::streamsize>(payload.size()));
    destination.flush();
  } catch (...) {
    return false;
  }
  return destination.good();
}

void write_help(std::ostream& output) {
  output
      << "TensorKiln bounded workload CLI\n"
      << "\n"
      << "Usage:\n"
      << "  tensorkiln list [--format text|json]\n"
      << "  tensorkiln inspect --workload ID [--format text|json]\n"
      << "  tensorkiln --help\n"
      << "\n"
      << "The CLI constructs versioned, compiled-in workloads through the "
         "public\n"
      << "GraphBuilder and ExecutionPlanCompiler APIs. It is not a graph "
         "dump parser\n"
      << "or a model-file importer.\n"
      << "\n"
      << "Exit codes: 0 success, 2 usage error, 3 workload build failure, "
         "70 internal failure.\n";
}

[[nodiscard]] OutputFormat parse_format(const std::string_view value) {
  if (value == "text") {
    return OutputFormat::text;
  }
  if (value == "json") {
    return OutputFormat::json;
  }
  usage_failure("invalid_format",
                "--format must be either 'text' or 'json'");
}

struct CommandOptions final {
  OutputFormat format = OutputFormat::text;
  bool format_seen = false;
  std::string_view workload;
  bool workload_seen = false;
};

[[nodiscard]] std::string_view require_option_value(
    const std::span<const std::string_view> arguments, std::size_t& index,
    const std::string_view option) {
  if (index + 1U >= arguments.size()) {
    usage_failure("missing_option_value",
                  std::string(option) + " requires a value");
  }
  ++index;
  if (arguments[index].empty() || arguments[index].starts_with("--")) {
    usage_failure("missing_option_value",
                  std::string(option) + " requires a value");
  }
  return arguments[index];
}

[[nodiscard]] CommandOptions parse_options(
    const std::span<const std::string_view> arguments,
    const bool allow_workload) {
  CommandOptions options;
  for (std::size_t index = 0U; index < arguments.size(); ++index) {
    const std::string_view argument = arguments[index];
    if (argument == "--format") {
      if (options.format_seen) {
        usage_failure("duplicate_option", "--format was provided twice");
      }
      options.format_seen = true;
      options.format =
          parse_format(require_option_value(arguments, index, "--format"));
      continue;
    }
    if (argument.starts_with("--format=")) {
      if (options.format_seen) {
        usage_failure("duplicate_option", "--format was provided twice");
      }
      options.format_seen = true;
      const std::string_view value = argument.substr(9U);
      if (value.empty()) {
        usage_failure("missing_option_value",
                      "--format requires a value");
      }
      options.format = parse_format(value);
      continue;
    }
    if (argument == "--workload") {
      if (!allow_workload) {
        usage_failure("unexpected_option",
                      "--workload is not valid for this command");
      }
      if (options.workload_seen) {
        usage_failure("duplicate_option", "--workload was provided twice");
      }
      options.workload_seen = true;
      options.workload =
          require_option_value(arguments, index, "--workload");
      continue;
    }
    if (argument.starts_with("--workload=")) {
      if (!allow_workload) {
        usage_failure("unexpected_option",
                      "--workload is not valid for this command");
      }
      if (options.workload_seen) {
        usage_failure("duplicate_option", "--workload was provided twice");
      }
      options.workload_seen = true;
      options.workload = argument.substr(11U);
      if (options.workload.empty()) {
        usage_failure("missing_option_value",
                      "--workload requires a value");
      }
      continue;
    }
    if (argument == "--help") {
      usage_failure("help_position",
                    "--help is accepted only as the sole argument");
    }
    if (argument.starts_with("-")) {
      usage_failure("unknown_option",
                    "unknown option '" + std::string(argument) + "'");
    }
    usage_failure("unexpected_argument",
                  "unexpected positional argument '" +
                      std::string(argument) + "'");
  }
  return options;
}

void require_known_workload(const CommandOptions& options) {
  if (!options.workload_seen) {
    usage_failure("missing_workload",
                  "inspect requires --workload " +
                      std::string(kWorkloadId));
  }
  if (options.workload != kWorkloadId) {
    usage_failure("unknown_workload",
                  "unknown workload '" + std::string(options.workload) + "'");
  }
}

void write_workload_json(std::ostream& output) {
  output << "{\"id\":";
  write_json_string(output, kWorkloadId);
  output << ",\"kind\":\"compiled_in\",\"description\":";
  write_json_string(output, kWorkloadDescription);
  output
      << ",\"inputs\":[{\"name\":\"x\",\"dtype\":\"f32\","
         "\"shape\":[2,3],\"elements\":6}],"
      << "\"outputs\":[{\"name\":\"result\",\"dtype\":\"f32\","
         "\"shape\":[2,2],\"elements\":4}]}";
}

void write_list(const OutputFormat format, std::ostream& output) {
  if (format == OutputFormat::text) {
    output << "TensorKiln compiled-in workloads\n"
           << "  " << kWorkloadId << "  " << kWorkloadDescription << '\n'
           << "scope: bounded examples; no graph or model-file import\n";
    return;
  }
  output << "{\"schema\":";
  write_json_string(output, kListSchema);
  output << ",\"workloads\":[";
  write_workload_json(output);
  output << "]}\n";
}

void write_plan_stats_json(std::ostream& output,
                           const ExecutionPlanStats& stats) {
  output << "{\"values\":" << stats.value_count
         << ",\"inputs\":" << stats.input_count
         << ",\"constants\":" << stats.constant_count
         << ",\"steps\":" << stats.step_count
         << ",\"outputs\":" << stats.output_count
         << ",\"constant_bytes\":" << stats.owned_constant_bytes
         << ",\"scalar_steps\":" << stats.scalar_steps
         << ",\"workspace_bytes\":" << stats.workspace_bytes << '}';
}

void write_kernels_json(std::ostream& output,
                        const ExecutionPlan& plan) {
  output.put('[');
  for (std::size_t index = 0U; index < plan.steps().size(); ++index) {
    const ExecutionStep& step = plan.steps()[index];
    if (index != 0U) {
      output.put(',');
    }
    output << "{\"step\":" << step.ordinal()
           << ",\"source_node\":" << step.source_node().ordinal()
           << ",\"kind\":";
    write_json_string(output, dense_kernel_kind_name(step.kernel()));
    output << ",\"scalar_steps\":" << step.scalar_steps() << '}';
  }
  output.put(']');
}

void write_inspect_text(const ExecutionPlan& plan, std::ostream& output) {
  const ExecutionPlanStats& stats = plan.stats();
  output << "TensorKiln plan inspection\n"
         << "schema: " << kInspectSchema << '\n'
         << "workload: " << kWorkloadId << '\n'
         << "scope: compiled-in workload; not a model-file import\n"
         << "input: x f32[2,3] elements=6\n"
         << "output: result f32[2,2] elements=4\n"
         << "plan: values=" << stats.value_count
         << " steps=" << stats.step_count
         << " scalar_steps=" << stats.scalar_steps
         << " workspace_bytes=" << stats.workspace_bytes << '\n'
         << "kernels:";
  for (const ExecutionStep& step : plan.steps()) {
    output << (step.ordinal() == 0U ? " " : " -> ")
           << dense_kernel_kind_name(step.kernel());
  }
  output << "\n\n" << plan.dump();
}

void write_inspect_json(const ExecutionPlan& plan, std::ostream& output) {
  output << "{\"schema\":";
  write_json_string(output, kInspectSchema);
  output << ",\"workload\":";
  write_workload_json(output);
  output << ",\"plan\":{\"stats\":";
  write_plan_stats_json(output, plan.stats());
  output << ",\"kernels\":";
  write_kernels_json(output, plan);
  output << ",\"canonical_dump\":";
  const std::string dump = plan.dump();
  write_json_string(output, dump);
  output << "}}\n";
}

[[nodiscard]] int run_impl(
    const std::span<const std::string_view> arguments,
    std::ostream& output) {
  validate_argument_envelope(arguments);
  if (arguments.empty()) {
    usage_failure("missing_command",
                  "a command is required; use 'tensorkiln --help'");
  }
  if (arguments.size() == 1U &&
      (arguments[0] == "--help" || arguments[0] == "-h")) {
    write_help(output);
    return kExitSuccess;
  }

  const std::string_view command = arguments[0];
  const std::span<const std::string_view> tail = arguments.subspan(1U);
  if (command == "list") {
    const CommandOptions options = parse_options(tail, false);
    write_list(options.format, output);
    return kExitSuccess;
  }
  if (command == "inspect") {
    const CommandOptions options = parse_options(tail, true);
    require_known_workload(options);
    const ExecutionPlan plan = build_dense_relu_plan();
    if (options.format == OutputFormat::text) {
      write_inspect_text(plan, output);
    } else {
      write_inspect_json(plan, output);
    }
    return kExitSuccess;
  }
  if (command.starts_with("-")) {
    usage_failure("unknown_option",
                  "unknown top-level option '" + std::string(command) + "'");
  }
  usage_failure("unknown_command",
                "unknown command '" + std::string(command) + "'");
}

}  // namespace

int run(const std::span<const std::string_view> arguments,
        std::ostream& output, std::ostream& error) {
  const bool json_errors = requests_json_errors(arguments);
  try {
    std::ostringstream staged_output = make_canonical_stream();
    const int status = run_impl(arguments, staged_output);
    if (publish(output, staged_output.str())) {
      return status;
    }
    const CliFailure failure(
        kExitInternalFailure, "output_write_failed",
        "command output could not be written completely");
    std::ostringstream staged_error = make_canonical_stream();
    write_error(staged_error, failure, json_errors);
    static_cast<void>(publish(error, staged_error.str()));
    return kExitInternalFailure;
  } catch (const CliFailure& failure) {
    std::ostringstream staged_error = make_canonical_stream();
    write_error(staged_error, failure, json_errors);
    if (publish(error, staged_error.str())) {
      return failure.exit_code();
    }
    return kExitInternalFailure;
  }
}

}  // namespace tensorkiln::cli
