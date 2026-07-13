CXX ?= g++
AR ?= ar
PROFILE ?= debug

CXX_TAG := $(notdir $(firstword $(CXX)))
BUILD_DIR := build/$(CXX_TAG)/$(PROFILE)

LIB_SOURCES := src/diagnostic.cpp src/graph.cpp src/shape.cpp \
               src/shape_inference.cpp src/tensor_type.cpp
TEST_SOURCES := tests/test_graph.cpp tests/test_main.cpp tests/test_result.cpp \
                tests/test_shape.cpp tests/test_shape_inference.cpp \
                tests/test_tensor_type.cpp
EXAMPLE_SOURCES := examples/inspect_graph.cpp

LIB_OBJECTS := $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(LIB_SOURCES))
TEST_OBJECTS := $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(TEST_SOURCES))
EXAMPLE_OBJECTS := $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(EXAMPLE_SOURCES))
DEPENDENCIES := $(LIB_OBJECTS:.o=.d) $(TEST_OBJECTS:.o=.d) \
                $(EXAMPLE_OBJECTS:.o=.d)

LIBRARY := $(BUILD_DIR)/libtensorkiln.a
TEST_BINARY := $(BUILD_DIR)/tensorkiln_tests
EXAMPLE_BINARY := $(BUILD_DIR)/inspect_graph

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

.PHONY: all test check sanitize oracle example clean help

all: $(LIBRARY) $(EXAMPLE_BINARY)

test: $(TEST_BINARY) $(EXAMPLE_BINARY)
	$(TEST_BINARY)

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

example: $(EXAMPLE_BINARY)
	$(EXAMPLE_BINARY)

$(LIBRARY): $(LIB_OBJECTS)
	@mkdir -p $(dir $@)
	$(AR) $(ARFLAGS) $@ $^

$(TEST_BINARY): $(TEST_OBJECTS) $(LIBRARY)
	@mkdir -p $(dir $@)
	$(CXX) $(TEST_OBJECTS) $(LIBRARY) $(LDFLAGS) $(PROFILE_LDFLAGS) -o $@

$(EXAMPLE_BINARY): $(EXAMPLE_OBJECTS) $(LIBRARY)
	@mkdir -p $(dir $@)
	$(CXX) $(EXAMPLE_OBJECTS) $(LIBRARY) $(LDFLAGS) $(PROFILE_LDFLAGS) -o $@

$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(PROJECT_CPPFLAGS) $(CPPFLAGS) $(PROJECT_CXXFLAGS) \
	       $(PROFILE_CXXFLAGS) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf build

help:
	@echo 'Targets: all test check sanitize oracle example clean help'
	@echo 'Profiles: debug (default), release, sanitize'
	@echo 'Example: make -j2 CXX=g++ PROFILE=release test'

-include $(DEPENDENCIES)
