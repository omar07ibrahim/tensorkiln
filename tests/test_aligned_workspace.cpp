#include "test.hpp"

#include <cstddef>
#include <cstdint>

#include "../src/aligned_workspace.hpp"
#include "tensorkiln/arena.hpp"

namespace {

using tensorkiln::detail::AlignedWorkspace;

TK_TEST("Aligned execution workspace owns exact logical bytes and guards") {
  AlignedWorkspace workspace(128U);
  TK_REQUIRE_EQ(workspace.logical_bytes(), 128U);
  TK_REQUIRE_EQ(workspace.bytes().size(), 128U);
  TK_REQUIRE(reinterpret_cast<std::uintptr_t>(workspace.bytes().data()) %
                 tensorkiln::kArenaAlignmentBytes ==
             0U);
  TK_REQUIRE(workspace.guards_intact());
  workspace.bytes().front() = std::byte{0x11U};
  workspace.bytes().back() = std::byte{0x22U};
  TK_REQUIRE(workspace.guards_intact());
}

TK_TEST("Aligned execution workspace detects both outer guard faults") {
  AlignedWorkspace prefix(64U);
  prefix.corrupt_prefix_for_test();
  TK_REQUIRE(!prefix.guards_intact());

  AlignedWorkspace suffix(64U);
  suffix.corrupt_suffix_for_test();
  TK_REQUIRE(!suffix.guards_intact());
}

TK_TEST("Zero-byte execution workspace performs no physical allocation") {
  AlignedWorkspace workspace(0U);
  TK_REQUIRE_EQ(workspace.logical_bytes(), 0U);
  TK_REQUIRE(workspace.bytes().empty());
  TK_REQUIRE(workspace.guards_intact());
  workspace.corrupt_prefix_for_test();
  workspace.corrupt_suffix_for_test();
  TK_REQUIRE(workspace.guards_intact());
}

}  // namespace
