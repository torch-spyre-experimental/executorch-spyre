# Copyright 2026 The ExecuTorch-Spyre Authors.
# Licensed under the Apache License, Version 2.0.
#
# Top-level CMake for the executorch-spyre package.
# Delegates to backends/spyre and backends/spyre2 when their options are ON.
#
# Quick start (host-only unit tests, no Spyre device):
#   cmake -S . -B /tmp/spyre-build \
#     -DEXECUTORCH_BUILD_SPYRE=OFF -DEXECUTORCH_BUILD_SPYRE_TESTS=ON
#   cmake --build /tmp/spyre-build
#   ctest --test-dir /tmp/spyre-build --output-on-failure
#
# The ExecuTorch source tree is resolved in this order:
#   1. -DEXECUTORCH_ROOT=<path> passed on the cmake command line
#   2. The vendored third_party/executorch submodule (default for fresh clones)
# Override with -DEXECUTORCH_ROOT=<path> to point at an existing checkout.

cmake_minimum_required(VERSION 3.19)
project(executorch_spyre LANGUAGES C CXX)

# Resolve ExecuTorch root: submodule default, overridable from the command line.
if(NOT DEFINED EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT "${CMAKE_SOURCE_DIR}/third_party/executorch"
      CACHE PATH "ExecuTorch source tree (tools/cmake/Utils.cmake etc.)")
endif()

option(EXECUTORCH_BUILD_SPYRE        "Build the spyre (SDSC) backend"            OFF)
option(EXECUTORCH_BUILD_SPYRE_TESTS  "Build host-only unit tests for spyre"       OFF)
option(EXECUTORCH_BUILD_SPYRE2       "Build the spyre2 (Triton/KTIR) backend"     OFF)
option(EXECUTORCH_BUILD_SPYRE2_TESTS "Build host-only unit tests for spyre2"      OFF)

# Register tests at the top level too, so `ctest --test-dir <build>` finds the
# gtests that the backend subdirectories add.
if(EXECUTORCH_BUILD_SPYRE_TESTS OR EXECUTORCH_BUILD_SPYRE2_TESTS)
  enable_testing()
endif()

if(EXECUTORCH_BUILD_SPYRE OR EXECUTORCH_BUILD_SPYRE_TESTS)
  add_subdirectory(executorch/backends/spyre)
endif()

if(EXECUTORCH_BUILD_SPYRE2 OR EXECUTORCH_BUILD_SPYRE2_TESTS)
  add_subdirectory(executorch/backends/spyre2)
endif()
