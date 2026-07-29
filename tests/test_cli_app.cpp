#include "test.hpp"

#include <array>
#include <ios>
#include <locale>
#include <sstream>
#include <streambuf>
#include <string>
#include <string_view>

#include "cli_app.hpp"

namespace {

struct Invocation final {
  int status;
  std::string output;
  std::string error;
};

class GroupingPunctuation final : public std::numpunct<char> {
 protected:
  [[nodiscard]] char do_thousands_sep() const override { return '_'; }
  [[nodiscard]] std::string do_grouping() const override { return "\1"; }
};

class FailingBuffer final : public std::streambuf {
 protected:
  [[nodiscard]] std::streamsize xsputn(
      const char*, std::streamsize) override {
    return 0;
  }

  [[nodiscard]] int_type overflow(int_type) override {
    return traits_type::eof();
  }

  [[nodiscard]] int sync() override { return -1; }
};

template <std::size_t Size>
[[nodiscard]] Invocation invoke(
    const std::array<std::string_view, Size>& arguments) {
  std::ostringstream output;
  std::ostringstream error;
  const int status =
      tensorkiln::cli::run(arguments, output, error);
  return Invocation{status, output.str(), error.str()};
}

}  // namespace

TK_TEST("CLI help states its bounded workload boundary") {
  constexpr std::array<std::string_view, 1U> arguments{{"--help"}};
  const Invocation invocation = invoke(arguments);

  TK_REQUIRE_EQ(invocation.status, tensorkiln::cli::kExitSuccess);
  TK_REQUIRE(invocation.error.empty());
  TK_REQUIRE(
      invocation.output.find("TensorKiln bounded workload CLI\n") == 0U);
  TK_REQUIRE(
      invocation.output.find("not a graph dump parser") != std::string::npos);
}

TK_TEST("CLI list output is stable in text and JSON formats") {
  constexpr std::array<std::string_view, 1U> text_arguments{{"list"}};
  constexpr std::array<std::string_view, 2U> json_arguments{{
      "list",
      "--format=json",
  }};

  const Invocation text = invoke(text_arguments);
  const Invocation first_json = invoke(json_arguments);
  const Invocation second_json = invoke(json_arguments);

  TK_REQUIRE_EQ(text.status, tensorkiln::cli::kExitSuccess);
  TK_REQUIRE(text.error.empty());
  TK_REQUIRE(
      text.output.find("dense_relu_v1") != std::string::npos);
  TK_REQUIRE_EQ(first_json.status, tensorkiln::cli::kExitSuccess);
  TK_REQUIRE(first_json.error.empty());
  TK_REQUIRE_EQ(first_json.output, second_json.output);
  TK_REQUIRE(
      first_json.output.find("\"schema\":\"tensorkiln.cli.workloads.v1\"") !=
      std::string::npos);
  TK_REQUIRE(
      first_json.output.find("\"kind\":\"compiled_in\"") !=
      std::string::npos);
}

TK_TEST("CLI inspect reports the real compiled plan deterministically") {
  constexpr std::array<std::string_view, 4U> arguments{{
      "inspect",
      "--workload",
      "dense_relu_v1",
      "--format=json",
  }};

  const Invocation first = invoke(arguments);
  const Invocation second = invoke(arguments);

  TK_REQUIRE_EQ(first.status, tensorkiln::cli::kExitSuccess);
  TK_REQUIRE(first.error.empty());
  TK_REQUIRE_EQ(first.output, second.output);
  TK_REQUIRE(
      first.output.find("\"schema\":\"tensorkiln.cli.inspect.v1\"") !=
      std::string::npos);
  TK_REQUIRE(first.output.find(
                 "\"values\":6,\"inputs\":1,\"constants\":2,"
                 "\"steps\":3,\"outputs\":1,\"constant_bytes\":32,"
                 "\"scalar_steps\":20,\"workspace_bytes\":128") !=
             std::string::npos);
  TK_REQUIRE(
      first.output.find("\"kind\":\"matmul_rank2_f32\"") !=
      std::string::npos);
  TK_REQUIRE(
      first.output.find("\"kind\":\"add_broadcast_f32\"") !=
      std::string::npos);
  TK_REQUIRE(
      first.output.find("\"kind\":\"relu_contiguous_f32\"") !=
      std::string::npos);
  TK_REQUIRE(
      first.output.find("tensorkiln.execution_plan v0 {\\n") !=
      std::string::npos);
}

TK_TEST("CLI text inspection exposes plan and kernel sequence") {
  constexpr std::array<std::string_view, 3U> arguments{{
      "inspect",
      "--workload=dense_relu_v1",
      "--format",
  }};
  const Invocation missing_value = invoke(arguments);
  TK_REQUIRE_EQ(missing_value.status, tensorkiln::cli::kExitUsage);
  TK_REQUIRE(missing_value.output.empty());
  TK_REQUIRE(
      missing_value.error.find("missing_option_value") != std::string::npos);

  constexpr std::array<std::string_view, 3U> valid_arguments{{
      "inspect",
      "--workload",
      "dense_relu_v1",
  }};
  const Invocation valid = invoke(valid_arguments);
  TK_REQUIRE_EQ(valid.status, tensorkiln::cli::kExitSuccess);
  TK_REQUIRE(valid.error.empty());
  TK_REQUIRE(
      valid.output.find("schema: tensorkiln.cli.inspect.v1") !=
      std::string::npos);
  TK_REQUIRE(
      valid.output.find(
          "kernels: matmul_rank2_f32 -> add_broadcast_f32 -> "
          "relu_contiguous_f32") != std::string::npos);
  TK_REQUIRE(
      valid.output.find("tensorkiln.execution_plan v0 {") !=
      std::string::npos);
}

TK_TEST("CLI rejects missing and unknown workloads without stdout") {
  constexpr std::array<std::string_view, 1U> missing_arguments{{
      "inspect",
  }};
  constexpr std::array<std::string_view, 3U> unknown_arguments{{
      "inspect",
      "--workload",
      "unknown",
  }};

  const Invocation missing = invoke(missing_arguments);
  const Invocation unknown = invoke(unknown_arguments);

  TK_REQUIRE_EQ(missing.status, tensorkiln::cli::kExitUsage);
  TK_REQUIRE(missing.output.empty());
  TK_REQUIRE(
      missing.error.find("missing_workload") != std::string::npos);
  TK_REQUIRE_EQ(unknown.status, tensorkiln::cli::kExitUsage);
  TK_REQUIRE(unknown.output.empty());
  TK_REQUIRE(
      unknown.error.find("unknown_workload") != std::string::npos);
}

TK_TEST("CLI renders versioned JSON errors on request") {
  constexpr std::array<std::string_view, 4U> arguments{{
      "inspect",
      "--workload",
      "missing",
      "--format=json",
  }};
  const Invocation invocation = invoke(arguments);

  TK_REQUIRE_EQ(invocation.status, tensorkiln::cli::kExitUsage);
  TK_REQUIRE(invocation.output.empty());
  TK_REQUIRE_EQ(
      invocation.error,
      "{\"schema\":\"tensorkiln.cli.error.v1\",\"error\":{\"code\":"
      "\"unknown_workload\",\"message\":\"unknown workload "
      "'missing'\"}}\n");
}

TK_TEST("CLI option parser rejects duplicate and stray arguments") {
  constexpr std::array<std::string_view, 5U> duplicate_arguments{{
      "inspect",
      "--workload=dense_relu_v1",
      "--workload",
      "dense_relu_v1",
      "--format=json",
  }};
  constexpr std::array<std::string_view, 2U> stray_arguments{{
      "list",
      "dense_relu_v1",
  }};

  const Invocation duplicate = invoke(duplicate_arguments);
  const Invocation stray = invoke(stray_arguments);

  TK_REQUIRE_EQ(duplicate.status, tensorkiln::cli::kExitUsage);
  TK_REQUIRE(duplicate.output.empty());
  TK_REQUIRE(
      duplicate.error.find("\"code\":\"duplicate_option\"") !=
      std::string::npos);
  TK_REQUIRE_EQ(stray.status, tensorkiln::cli::kExitUsage);
  TK_REQUIRE(stray.output.empty());
  TK_REQUIRE(
      stray.error.find("unexpected_argument") != std::string::npos);

  constexpr std::array<std::string_view, 2U> empty_format_arguments{{
      "list",
      "--format=",
  }};
  constexpr std::array<std::string_view, 2U> short_option_arguments{{
      "list",
      "-x",
  }};
  const Invocation empty_format = invoke(empty_format_arguments);
  const Invocation short_option = invoke(short_option_arguments);
  TK_REQUIRE(
      empty_format.error.find("missing_option_value") !=
      std::string::npos);
  TK_REQUIRE(
      short_option.error.find("unknown_option") != std::string::npos);
}

TK_TEST("CLI rejects unknown commands and top-level options distinctly") {
  constexpr std::array<std::string_view, 1U> command_arguments{{
      "compile",
  }};
  constexpr std::array<std::string_view, 1U> option_arguments{{
      "--version",
  }};

  const Invocation command = invoke(command_arguments);
  const Invocation option = invoke(option_arguments);

  TK_REQUIRE_EQ(command.status, tensorkiln::cli::kExitUsage);
  TK_REQUIRE(command.error.find("unknown_command") != std::string::npos);
  TK_REQUIRE_EQ(option.status, tensorkiln::cli::kExitUsage);
  TK_REQUIRE(option.error.find("unknown_option") != std::string::npos);
}

TK_TEST("CLI bounds and ASCII-validates its argument envelope") {
  const std::string oversized(8193U, 'x');
  const std::array<std::string_view, 2U> oversized_arguments{{
      "list",
      oversized,
  }};
  const std::string non_ascii(1U, static_cast<char>(0xff));
  const std::array<std::string_view, 2U> non_ascii_arguments{{
      "list",
      non_ascii,
  }};

  const Invocation oversized_result = invoke(oversized_arguments);
  const Invocation non_ascii_result = invoke(non_ascii_arguments);

  TK_REQUIRE_EQ(oversized_result.status, tensorkiln::cli::kExitUsage);
  TK_REQUIRE(oversized_result.output.empty());
  TK_REQUIRE(
      oversized_result.error.find("argument_limit_exceeded") !=
      std::string::npos);
  TK_REQUIRE_EQ(non_ascii_result.status, tensorkiln::cli::kExitUsage);
  TK_REQUIRE(non_ascii_result.output.empty());
  TK_REQUIRE(
      non_ascii_result.error.find("invalid_argument_encoding") !=
      std::string::npos);
}

TK_TEST("CLI JSON ignores caller locale and formatting flags") {
  constexpr std::array<std::string_view, 4U> arguments{{
      "inspect",
      "--workload",
      "dense_relu_v1",
      "--format=json",
  }};
  const Invocation canonical = invoke(arguments);

  std::ostringstream polluted_output;
  polluted_output.imbue(
      std::locale(std::locale::classic(), new GroupingPunctuation));
  polluted_output.setf(std::ios_base::hex, std::ios_base::basefield);
  polluted_output.setf(std::ios_base::showbase);
  polluted_output.setf(std::ios_base::showpos);
  std::ostringstream error;
  const int status =
      tensorkiln::cli::run(arguments, polluted_output, error);

  TK_REQUIRE_EQ(status, tensorkiln::cli::kExitSuccess);
  TK_REQUIRE(error.str().empty());
  TK_REQUIRE_EQ(polluted_output.str(), canonical.output);
}

TK_TEST("CLI reports a failed stdout write as an internal failure") {
  constexpr std::array<std::string_view, 2U> arguments{{
      "list",
      "--format=json",
  }};
  FailingBuffer failing_buffer;
  std::ostream failing_output(&failing_buffer);
  std::ostringstream error;

  const int status =
      tensorkiln::cli::run(arguments, failing_output, error);

  TK_REQUIRE_EQ(status, tensorkiln::cli::kExitInternalFailure);
  TK_REQUIRE(
      error.str().find("\"code\":\"output_write_failed\"") !=
      std::string::npos);
}

TK_TEST("CLI returns internal failure when stderr cannot be written") {
  constexpr std::array<std::string_view, 1U> arguments{{
      "missing-command",
  }};
  std::ostringstream output;
  FailingBuffer failing_buffer;
  std::ostream failing_error(&failing_buffer);

  const int status =
      tensorkiln::cli::run(arguments, output, failing_error);

  TK_REQUIRE_EQ(status, tensorkiln::cli::kExitInternalFailure);
  TK_REQUIRE(output.str().empty());
}
