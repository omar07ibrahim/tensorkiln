CXX ?= g++
AR ?= ar
PROFILE ?= debug

CXX_TAG := $(notdir $(firstword $(CXX)))
BUILD_DIR := build/$(CXX_TAG)/$(PROFILE)

LIB_SOURCES := src/arena.cpp src/arena_planner.cpp src/arena_support.cpp \
               src/arena_verifier.cpp src/compiler_support.cpp \
               src/dead_code_elimination.cpp \
               src/diagnostic.cpp src/aligned_workspace.cpp \
               src/execution.cpp src/execution_kernels.cpp \
               src/execution_plan.cpp \
               src/execution_plan_compiler.cpp \
               src/execution_plan_support.cpp \
               src/execution_plan_verifier.cpp src/graph.cpp src/graph_arena.cpp \
               src/graph_arena_lowering.cpp \
               src/graph_arena_verifier.cpp src/provenance.cpp \
               src/reference.cpp src/shape.cpp src/shape_inference.cpp \
               src/structural_canonicalization.cpp src/tensor_type.cpp
TEST_SOURCES := tests/test_arena_planner.cpp tests/test_arena_seeded.cpp \
                tests/test_arena_verifier.cpp \
                tests/test_aligned_workspace.cpp \
                tests/test_dead_code_elimination.cpp \
                tests/test_execution.cpp tests/test_execution_seeded.cpp \
                tests/test_graph.cpp \
                tests/test_execution_plan.cpp \
                tests/test_execution_plan_verifier.cpp \
                tests/test_graph_arena.cpp \
                tests/test_graph_arena_seeded.cpp \
                tests/test_main.cpp tests/test_reference.cpp \
                tests/test_provenance.cpp tests/test_result.cpp \
                tests/test_shape.cpp tests/test_shape_inference.cpp \
                tests/test_structural_canonicalization.cpp \
                tests/test_structural_canonicalization_contracts.cpp \
                tests/test_structural_canonicalization_seeded.cpp \
                tests/test_tensor_type.cpp
EXAMPLE_SOURCES := examples/inspect_graph.cpp examples/plan_arena.cpp \
                   examples/execute_graph.cpp
NOALLOC_SOURCE := tests/execution_noalloc_main.cpp

LIB_OBJECTS := $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(LIB_SOURCES))
TEST_OBJECTS := $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(TEST_SOURCES))
EXAMPLE_OBJECTS := $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(EXAMPLE_SOURCES))
NOALLOC_OBJECT := $(BUILD_DIR)/$(NOALLOC_SOURCE:.cpp=.o)
DEPENDENCIES := $(LIB_OBJECTS:.o=.d) $(TEST_OBJECTS:.o=.d) \
                $(EXAMPLE_OBJECTS:.o=.d) $(NOALLOC_OBJECT:.o=.d)

LIBRARY := $(BUILD_DIR)/libtensorkiln.a
TEST_BINARY := $(BUILD_DIR)/tensorkiln_tests
EXAMPLE_BINARIES := $(patsubst examples/%.cpp,$(BUILD_DIR)/%,\
                    $(EXAMPLE_SOURCES))
NOALLOC_BINARY := $(BUILD_DIR)/execution_noalloc
VISUALS_TEST := tests/test_readme_visuals.py
VISUALS_TOOL := tools/render_readme_visuals.py
NOALLOC_WRAP_LDFLAGS := -Wl,--wrap=malloc -Wl,--wrap=calloc \
                         -Wl,--wrap=realloc -Wl,--wrap=aligned_alloc \
                         -Wl,--wrap=posix_memalign

TEST_AUDIT_BINARY :=
ifeq ($(PROFILE),release)
  TEST_AUDIT_BINARY := $(NOALLOC_BINARY)
endif

PROJECT_CPPFLAGS := -Iinclude -Itests -MMD -MP
PROJECT_CXXFLAGS := -std=c++20 -Wall -Wextra -Wpedantic -Werror \
                    -Wconversion -Wsign-conversion -Wshadow -Wcast-qual \
                    -Wformat=2 -Wundef -Wnon-virtual-dtor -Wold-style-cast \
                    -Woverloaded-virtual -Wdouble-promotion \
                    -Wimplicit-fallthrough -Wswitch-enum \
                    -Wzero-as-null-pointer-constant -Wdate-time \
                    -fno-fast-math -ffp-contract=off \
                    -ffile-prefix-map=$(CURDIR)=.

ifeq ($(PROFILE),debug)
  PROFILE_CXXFLAGS := -O0 -g3 -D_GLIBCXX_ASSERTIONS -fno-omit-frame-pointer
  PROFILE_LDFLAGS :=
else ifeq ($(PROFILE),release)
  PROFILE_CXXFLAGS := -O3 -DNDEBUG
  PROFILE_LDFLAGS :=
else ifeq ($(PROFILE),sanitize)
  PROFILE_CXXFLAGS := -O1 -g3 -D_GLIBCXX_ASSERTIONS -fno-omit-frame-pointer \
                      -fsanitize=address,undefined -fno-sanitize-recover=all
  PROFILE_LDFLAGS := -fsanitize=address,undefined -fno-sanitize-recover=all
else
  $(error unsupported PROFILE '$(PROFILE)'; use debug, release, or sanitize)
endif

ARFLAGS := rcsD

.PHONY: all test noalloc check sanitize oracle example run-examples visuals \
        visuals-check visuals-generate visuals-verify clean help

all: $(LIBRARY) $(EXAMPLE_BINARIES)

test: $(TEST_BINARY) run-examples $(TEST_AUDIT_BINARY)
	$(TEST_BINARY)
ifeq ($(PROFILE),release)
	$(NOALLOC_BINARY)
	python3 -B -I $(VISUALS_TEST)
	python3 -B -I $(VISUALS_TOOL) --build-dir $(BUILD_DIR) --check
endif

noalloc: $(NOALLOC_BINARY)
	$(NOALLOC_BINARY)

check:
	$(MAKE) PROFILE=debug test
	$(MAKE) PROFILE=release test
	$(MAKE) oracle

sanitize:
	ASAN_OPTIONS=detect_leaks=1:halt_on_error=1:strict_string_checks=1 \
	UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
	$(MAKE) PROFILE=sanitize test

oracle:
	python3 -I tools/oracle.py --check tests/oracle_fixture.hpp

example: run-examples

run-examples: $(EXAMPLE_BINARIES)
	$(BUILD_DIR)/inspect_graph
	$(BUILD_DIR)/plan_arena
	$(BUILD_DIR)/execute_graph

visuals:
	$(MAKE) PROFILE=release visuals-generate

visuals-check:
	$(MAKE) PROFILE=release visuals-verify

visuals-generate: $(EXAMPLE_BINARIES)
	python3 -B -I $(VISUALS_TOOL) --build-dir $(BUILD_DIR)

visuals-verify: $(EXAMPLE_BINARIES)
	python3 -B -I $(VISUALS_TEST)
	python3 -B -I $(VISUALS_TOOL) --build-dir $(BUILD_DIR) --check

$(LIBRARY): $(LIB_OBJECTS)
	@mkdir -p $(dir $@)
	$(AR) $(ARFLAGS) $@ $^

$(TEST_BINARY): $(TEST_OBJECTS) $(LIBRARY)
	@mkdir -p $(dir $@)
	$(CXX) $(TEST_OBJECTS) $(LIBRARY) $(LDFLAGS) $(PROFILE_LDFLAGS) -o $@

$(NOALLOC_BINARY): $(NOALLOC_OBJECT) $(LIBRARY)
	@mkdir -p $(dir $@)
	$(CXX) $(NOALLOC_OBJECT) $(LIBRARY) $(LDFLAGS) $(PROFILE_LDFLAGS) \
	       $(NOALLOC_WRAP_LDFLAGS) -o $@

$(EXAMPLE_BINARIES): $(BUILD_DIR)/%: $(BUILD_DIR)/examples/%.o $(LIBRARY)
	@mkdir -p $(dir $@)
	$(CXX) $< $(LIBRARY) $(LDFLAGS) $(PROFILE_LDFLAGS) -o $@

$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(PROJECT_CPPFLAGS) $(CPPFLAGS) $(PROJECT_CXXFLAGS) \
	       $(PROFILE_CXXFLAGS) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf build

help:
	@echo 'Targets: all test noalloc check sanitize oracle example visuals visuals-check clean help'
	@echo 'Profiles: debug (default), release, sanitize'
	@echo 'Example: make -j2 CXX=g++ PROFILE=release test'

-include $(DEPENDENCIES)
