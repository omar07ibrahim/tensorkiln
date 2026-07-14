#include "test.hpp"

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "tensorkiln/provenance.hpp"

namespace {

using tensorkiln::ErrorCode;
using tensorkiln::GraphBuilder;
using tensorkiln::GraphProvenance;
using tensorkiln::NodeId;
using tensorkiln::NodeProvenance;
using tensorkiln::OutputId;
using tensorkiln::Shape;
using tensorkiln::TensorType;
using tensorkiln::ValueId;
using tensorkiln::VerifiedGraph;

[[nodiscard]] TensorType scalar_type() {
  const auto type = TensorType::create(Shape::scalar());
  TK_REQUIRE(type.value_if() != nullptr);
  return *type.value_if();
}

[[nodiscard]] ValueId require_value(
    const tensorkiln::Result<ValueId>& result) {
  TK_REQUIRE(result.value_if() != nullptr);
  return *result.value_if();
}

void require_output(const tensorkiln::Result<OutputId>& result) {
  TK_REQUIRE(result.value_if() != nullptr);
}

[[nodiscard]] VerifiedGraph require_graph(
    tensorkiln::Result<VerifiedGraph> result) {
  TK_REQUIRE(result.value_if() != nullptr);
  return std::move(*result.value_if());
}

[[nodiscard]] GraphProvenance require_provenance(
    tensorkiln::Result<GraphProvenance> result) {
  TK_REQUIRE(result.value_if() != nullptr);
  return std::move(*result.value_if());
}

template <typename T>
const tensorkiln::Diagnostic& require_error(
    const tensorkiln::Result<T>& result, const ErrorCode code) {
  TK_REQUIRE(result.error_if() != nullptr);
  TK_REQUIRE_EQ(result.error_if()->code, code);
  return *result.error_if();
}

[[nodiscard]] VerifiedGraph build_chain(const std::size_t node_count,
                                        const std::string& prefix) {
  TK_REQUIRE(node_count > 0U);
  GraphBuilder builder;
  ValueId value =
      require_value(builder.input(prefix + "_input", scalar_type()));
  for (std::size_t index = 1U; index < node_count; ++index) {
    value = require_value(builder.relu(value));
  }
  require_output(builder.output(prefix + "_output", value));
  return require_graph(std::move(builder).finish());
}

TK_TEST("Provenance factory canonicalizes sources and supports full lookups") {
  const VerifiedGraph source = build_chain(4U, "source");
  const VerifiedGraph result = build_chain(2U, "result");
  const VerifiedGraph foreign_result = build_chain(2U, "foreign");

  std::vector<std::vector<NodeId>> sources_by_result{
      {source.nodes()[0].id()},
      {
          source.nodes()[3].id(),
          source.nodes()[1].id(),
          source.nodes()[2].id(),
          source.nodes()[1].id(),
      },
  };
  const GraphProvenance provenance = require_provenance(
      GraphProvenance::create(result, source, std::move(sources_by_result)));

  TK_REQUIRE_EQ(provenance.entries().size(), 2U);
  const NodeProvenance& first = provenance.entries()[0];
  const NodeProvenance& second = provenance.entries()[1];
  TK_REQUIRE_EQ(first.result_node(), result.nodes()[0].id());
  TK_REQUIRE_EQ(first.result_value(), result.nodes()[0].output());
  TK_REQUIRE_EQ(first.sources().size(), 1U);
  TK_REQUIRE_EQ(first.sources()[0].node(), source.nodes()[0].id());
  TK_REQUIRE_EQ(first.sources()[0].value(), source.nodes()[0].output());

  TK_REQUIRE_EQ(second.result_node(), result.nodes()[1].id());
  TK_REQUIRE_EQ(second.result_value(), result.nodes()[1].output());
  TK_REQUIRE_EQ(second.sources().size(), 3U);
  for (std::size_t index = 0U; index < second.sources().size(); ++index) {
    TK_REQUIRE_EQ(second.sources()[index].node(),
                  source.nodes()[index + 1U].id());
    TK_REQUIRE_EQ(second.sources()[index].value(),
                  source.nodes()[index + 1U].output());
  }

  TK_REQUIRE(provenance.for_result(result.nodes()[0].id()) == &first);
  TK_REQUIRE(provenance.for_result(result.nodes()[0].output()) == &first);
  TK_REQUIRE(provenance.for_result(result.nodes()[1].id()) == &second);
  TK_REQUIRE(provenance.for_result(result.nodes()[1].output()) == &second);
  TK_REQUIRE(provenance.for_source(source.nodes()[0].id()) == &first);
  TK_REQUIRE(provenance.for_source(source.nodes()[0].output()) == &first);
  TK_REQUIRE(provenance.for_source(source.nodes()[2].id()) == &second);
  TK_REQUIRE(provenance.for_source(source.nodes()[2].output()) == &second);
  TK_REQUIRE(provenance.for_result(foreign_result.nodes()[0].id()) == nullptr);
  TK_REQUIRE(provenance.for_result(foreign_result.nodes()[0].output()) ==
             nullptr);

  const std::string expected_dump =
      "tensorkiln.provenance v0 {\n"
      "  #n0 %0 <- {source #n0 %0}\n"
      "  #n1 %1 <- {source #n1 %1, source #n2 %2, source #n3 %3}\n"
      "}\n";
  TK_REQUIRE_EQ(provenance.dump(), expected_dump);
  TK_REQUIRE_EQ(provenance.dump(), provenance.dump());
}

TK_TEST("Provenance factory rejects a result-entry count mismatch") {
  const VerifiedGraph source = build_chain(2U, "source");
  const VerifiedGraph result = build_chain(2U, "result");
  std::vector<std::vector<NodeId>> sources_by_result{
      {source.nodes()[0].id()},
  };

  const auto created = GraphProvenance::create(
      result, source, std::move(sources_by_result));
  const tensorkiln::Diagnostic& error =
      require_error(created, ErrorCode::provenance_domain_mismatch);
  TK_REQUIRE_EQ(error.message,
                "provenance map has 1 result entries; expected 2");
}

TK_TEST("Provenance factory rejects an empty source set") {
  const VerifiedGraph source = build_chain(2U, "source");
  const VerifiedGraph result = build_chain(2U, "result");
  std::vector<std::vector<NodeId>> sources_by_result{
      {source.nodes()[0].id()},
      {},
  };

  const auto created = GraphProvenance::create(
      result, source, std::move(sources_by_result));
  const tensorkiln::Diagnostic& error =
      require_error(created, ErrorCode::provenance_domain_mismatch);
  TK_REQUIRE_EQ(error.message, "result #n1 has no source definitions");
}

TK_TEST("Provenance factory rejects a foreign source owner") {
  const VerifiedGraph source = build_chain(2U, "source");
  const VerifiedGraph result = build_chain(2U, "result");
  const VerifiedGraph foreign = build_chain(2U, "foreign");
  std::vector<std::vector<NodeId>> sources_by_result{
      {source.nodes()[0].id()},
      {foreign.nodes()[1].id()},
  };

  const auto created = GraphProvenance::create(
      result, source, std::move(sources_by_result));
  const tensorkiln::Diagnostic& error =
      require_error(created, ErrorCode::provenance_domain_mismatch);
  TK_REQUIRE_EQ(error.message,
                "source #n1 does not belong to the declared source graph");
}

TK_TEST("Provenance factory rejects one source assigned to two results") {
  const VerifiedGraph source = build_chain(2U, "source");
  const VerifiedGraph result = build_chain(2U, "result");
  std::vector<std::vector<NodeId>> sources_by_result{
      {source.nodes()[0].id()},
      {source.nodes()[0].id()},
  };

  const auto created = GraphProvenance::create(
      result, source, std::move(sources_by_result));
  const tensorkiln::Diagnostic& error =
      require_error(created, ErrorCode::provenance_domain_mismatch);
  TK_REQUIRE_EQ(error.message,
                "source #n0 is assigned to both #n0 and #n1");
}

TK_TEST("Provenance composition expands canonical roots and rejects wrong owner") {
  const VerifiedGraph root = build_chain(4U, "root");
  const VerifiedGraph middle = build_chain(3U, "middle");
  const VerifiedGraph result = build_chain(2U, "result");

  std::vector<std::vector<NodeId>> upstream_sources{
      {root.nodes()[0].id()},
      {root.nodes()[2].id(), root.nodes()[1].id()},
      {root.nodes()[3].id()},
  };
  const GraphProvenance upstream = require_provenance(
      GraphProvenance::create(middle, root, std::move(upstream_sources)));

  std::vector<std::vector<NodeId>> downstream_sources{
      {middle.nodes()[0].id()},
      {middle.nodes()[2].id(), middle.nodes()[1].id()},
  };
  const GraphProvenance downstream = require_provenance(
      GraphProvenance::create(result, middle, std::move(downstream_sources)));
  const GraphProvenance composed =
      require_provenance(downstream.compose(upstream));

  const std::string expected_dump =
      "tensorkiln.provenance v0 {\n"
      "  #n0 %0 <- {source #n0 %0}\n"
      "  #n1 %1 <- {source #n1 %1, source #n2 %2, source #n3 %3}\n"
      "}\n";
  TK_REQUIRE_EQ(composed.dump(), expected_dump);
  const NodeProvenance* second =
      composed.for_result(result.nodes()[1].id());
  TK_REQUIRE(second != nullptr);
  TK_REQUIRE_EQ(second->sources().size(), 3U);
  TK_REQUIRE(composed.for_source(root.nodes()[2].id()) == second);
  TK_REQUIRE(composed.for_source(root.nodes()[2].output()) == second);

  const VerifiedGraph foreign_middle = build_chain(3U, "foreign_middle");
  std::vector<std::vector<NodeId>> foreign_sources{
      {root.nodes()[0].id()},
      {root.nodes()[1].id(), root.nodes()[2].id()},
      {root.nodes()[3].id()},
  };
  const GraphProvenance foreign_upstream = require_provenance(
      GraphProvenance::create(foreign_middle, root,
                              std::move(foreign_sources)));
  const auto wrong_owner = downstream.compose(foreign_upstream);
  const tensorkiln::Diagnostic& error =
      require_error(wrong_owner, ErrorCode::provenance_domain_mismatch);
  TK_REQUIRE_EQ(error.message,
                "upstream provenance does not describe #n0 %0");
}

struct DetachedProvenance final {
  GraphProvenance provenance;
  NodeId result_node;
  ValueId result_value;
  NodeId first_source_node;
  ValueId first_source_value;
  NodeId second_source_node;
  ValueId second_source_value;
};

[[nodiscard]] DetachedProvenance make_detached_provenance() {
  const VerifiedGraph source = build_chain(2U, "source");
  const VerifiedGraph result = build_chain(1U, "result");
  std::vector<std::vector<NodeId>> sources_by_result{
      {source.nodes()[1].id(), source.nodes()[0].id()},
  };
  GraphProvenance provenance = require_provenance(
      GraphProvenance::create(result, source, std::move(sources_by_result)));

  return DetachedProvenance{
      std::move(provenance),
      result.nodes()[0].id(),
      result.nodes()[0].output(),
      source.nodes()[0].id(),
      source.nodes()[0].output(),
      source.nodes()[1].id(),
      source.nodes()[1].output(),
  };
}

TK_TEST("Provenance owns handles after source and result graph destruction") {
  const DetachedProvenance detached = make_detached_provenance();
  TK_REQUIRE_EQ(detached.provenance.entries().size(), 1U);
  const NodeProvenance* entry =
      detached.provenance.for_result(detached.result_node);
  TK_REQUIRE(entry != nullptr);
  TK_REQUIRE(detached.provenance.for_result(detached.result_value) == entry);
  TK_REQUIRE(detached.provenance.for_source(detached.first_source_node) ==
             entry);
  TK_REQUIRE(detached.provenance.for_source(detached.first_source_value) ==
             entry);
  TK_REQUIRE(detached.provenance.for_source(detached.second_source_node) ==
             entry);
  TK_REQUIRE(detached.provenance.for_source(detached.second_source_value) ==
             entry);
  TK_REQUIRE_EQ(entry->sources().size(), 2U);

  const std::string expected_dump =
      "tensorkiln.provenance v0 {\n"
      "  #n0 %0 <- {source #n0 %0, source #n1 %1}\n"
      "}\n";
  TK_REQUIRE_EQ(detached.provenance.dump(), expected_dump);
}

}  // namespace
