# Copyright 2026 The Torch-Spyre / ExecuTorch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Spyre ExecuTorch backend delegate (runtime side).
#
# Gated by EXECUTORCH_BUILD_SPYRE. In an upstream tree this lives at
# executorch/executorch/backends/spyre/CMakeLists.txt and the top-level CMakeLists adds:
#
#     option(EXECUTORCH_BUILD_SPYRE "Build the IBM Spyre backend" OFF)
#     if(EXECUTORCH_BUILD_SPYRE)
#       add_subdirectory(executorch/backends/spyre)
#     endif()
#
# Built standalone against an installed torch-spyre. The Spyre stack paths have NO
# defaults: they must come from the torch_spyre/_C.so that Python imports (derive them
# with scripts/spyre_env.sh; see docs/setup-reference.md, "The Spyre stack must match
# torch-spyre's _C.so").

cmake_minimum_required(VERSION 3.19)

# When configured standalone (not add_subdirectory'd by ExecuTorch), give the
# tree a project() so the toolchain is detected explicitly. Harmless when this
# is a subdirectory of a parent project.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(spyre_backend LANGUAGES C CXX)
endif()

# The device runtime library (spyre_backend) needs the Spyre dev stack and torch;
# the host-only unit tests (EXECUTORCH_BUILD_SPYRE_TESTS below) need neither. In an
# upstream tree the parent only add_subdirectory()s this when EXECUTORCH_BUILD_SPYRE
# is ON, so defaulting ON keeps the standalone Phase 2/3 build unchanged; a
# test-only build on a machine without the Spyre stack passes
# -DEXECUTORCH_BUILD_SPYRE=OFF -DEXECUTORCH_BUILD_SPYRE_TESTS=ON.
option(EXECUTORCH_BUILD_SPYRE "Build the IBM Spyre device backend" ON)

# ExecuTorch source tree. The top-level CMakeLists sets it to third_party/executorch;
# configured standalone from this directory, that checkout is three levels up.
if(NOT DEFINED EXECUTORCH_ROOT)
  set(EXECUTORCH_ROOT
      "${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/executorch"
      CACHE PATH "ExecuTorch source tree (for tools/cmake/Utils.cmake, third-party/json)")
endif()

if(EXECUTORCH_BUILD_SPYRE)

# executorch_target_link_options_shared_lib (used below to keep the static
# register_backend() ctor) lives in ExecuTorch's tools/cmake/Utils.cmake, which
# the installed executorch-config.cmake does NOT pull in. Include it from the ET
# source tree when the macro isn't already defined (e.g. standalone build).
if(NOT COMMAND executorch_target_link_options_shared_lib)
  include("${EXECUTORCH_ROOT}/tools/cmake/Utils.cmake")
endif()

# ---- The Spyre stack: derived from torch_spyre/_C.so, never defaulted --------
# The runner runs in the same process as, and calls into, _C.so, so it must link the
# same libflex / DeepTools / libspyre_comms that _C.so loads. A default pointing at
# /opt/ibm/spyre or /home/senuser produced a wrong build three times, so there is
# none. Each value is taken from, in this order:
#   1. -D<VAR>=...            (given on this command line)
#   2. a value already in this build dir's CMakeCache.txt  (printed as [cache])
#   3. the <VAR> environment variable
# and is printed with its source below, so a stale shell value is visible.
# A cached value that differs from the environment is stale (an earlier configure of
# this directory); that is an error of its own, reported before the consistency check
# against ldd, because it is the cause and the ldd mismatch only a symptom.
# scripts/spyre_env.sh prints all six, derived from `ldd <_C.so>`:
#   mapfile -t A < <(scripts/spyre_env.sh --cmake-args); cmake ... "${A[@]}"
set(_spyre_vars TORCH_SPYRE_C_SO TORCH_SPYRE_CSRC SPYRE_RUNTIME_DIR
                SPYRE_DEEPTOOLS_DIR SPYRE_SENLIB_DIR SPYRE_COMMS_DIR)
# At this point CMakeCache.txt on disk still holds what earlier configures wrote; a -D
# on this command line is already in the in-memory cache but not yet in the file. So a
# value that equals the one on disk was cached, and one that differs was given with -D.
set(_spyre_cache_file "${CMAKE_BINARY_DIR}/CMakeCache.txt")
set(_spyre_missing "")
set(_spyre_stale "")
foreach(_v ${_spyre_vars})
  set(_on_disk "")
  if(EXISTS "${_spyre_cache_file}")
    file(STRINGS "${_spyre_cache_file}" _line REGEX "^${_v}:[A-Z]+=" LIMIT_COUNT 1)
    if(_line MATCHES "^[^=]*=(.*)$")
      set(_on_disk "${CMAKE_MATCH_1}")
    endif()
  endif()
  if(DEFINED CACHE{${_v}} AND NOT "$CACHE{${_v}}" STREQUAL "")
    if(NOT "${_on_disk}" STREQUAL "" AND "$CACHE{${_v}}" STREQUAL "${_on_disk}")
      set(_src "cache")
      if(NOT "$ENV{${_v}}" STREQUAL "" AND NOT "$ENV{${_v}}" STREQUAL "$CACHE{${_v}}")
        list(APPEND _spyre_stale
             "  ${_v}\n      cached:      $CACHE{${_v}}\n      environment: $ENV{${_v}}")
      endif()
    else()
      set(_src "-D")
    endif()
  elseif(NOT "$ENV{${_v}}" STREQUAL "")
    set(${_v} "$ENV{${_v}}" CACHE STRING "from the environment" FORCE)
    set(_src "env")
  else()
    list(APPEND _spyre_missing ${_v})
    continue()
  endif()
  message(STATUS "Spyre stack: ${_v} = ${${_v}}   [${_src}]")
endforeach()
if(_spyre_stale)
  string(REPLACE ";" "\n" _spyre_stale_str "${_spyre_stale}")
  message(FATAL_ERROR
    "Spyre stack: stale value from an existing CMakeCache.txt; reconfigure with "
    "`cmake --fresh` or an empty build dir.\n"
    "${CMAKE_BINARY_DIR}/CMakeCache.txt holds a value that differs from the "
    "environment:\n${_spyre_stale_str}\n"
    "`cmake --fresh` needs CMake >= 3.24; with any version, "
    "`rm -rf \"\${RUN_BUILD:?}\"` (your build dir) gives an empty one.")
endif()
if(_spyre_missing)
  string(REPLACE ";" ", " _spyre_missing_str "${_spyre_missing}")
  message(FATAL_ERROR
    "Spyre stack not configured: ${_spyre_missing_str} not set (neither -D nor "
    "environment).\n"
    "These have no defaults on purpose: the runner must link the SAME libflex, "
    "DeepTools and libspyre_comms as the torch_spyre/_C.so that Python imports, and "
    "a guessed path (/opt/ibm/spyre, /home/senuser/...) has repeatedly produced a "
    "runner that does not link or load.\n"
    "Derive them from that _C.so (LD_LIBRARY_PATH must already let "
    "`import torch_spyre._C` work):\n"
    "  mapfile -t SPYRE_CMAKE_ARGS < <(scripts/spyre_env.sh --cmake-args)\n"
    "  cmake ... \"\${SPYRE_CMAKE_ARGS[@]}\"\n"
    "or  eval \"\$(scripts/spyre_env.sh)\" to export them for the environment "
    "fallback. See docs/setup-reference.md, \"The Spyre stack must match torch-spyre's _C.so\".")
endif()

if(NOT EXISTS "${TORCH_SPYRE_C_SO}")
  message(FATAL_ERROR "TORCH_SPYRE_C_SO does not exist: ${TORCH_SPYRE_C_SO}")
endif()
if(NOT EXISTS "${TORCH_SPYRE_CSRC}/prepare_kernel.h")
  message(FATAL_ERROR
    "TORCH_SPYRE_CSRC (${TORCH_SPYRE_CSRC}) has no prepare_kernel.h.")
endif()
foreach(_v SPYRE_RUNTIME_DIR SPYRE_DEEPTOOLS_DIR SPYRE_SENLIB_DIR SPYRE_COMMS_DIR)
  if(NOT IS_DIRECTORY "${${_v}}/lib")
    message(FATAL_ERROR "${_v} (${${_v}}) has no lib/ directory.")
  endif()
endforeach()

# The ldd of _C.so is what this process will load at run time, i.e. it depends on
# LD_LIBRARY_PATH. It must agree with the -D / env values above, otherwise the runner
# is linked against one libflex and (via _C.so) loads another.
find_program(_spyre_ldd ldd)
if(NOT _spyre_ldd)
  message(FATAL_ERROR "ldd not found; cannot verify that ${TORCH_SPYRE_C_SO} resolves "
                      "the Spyre stack given by SPYRE_*_DIR.")
endif()
execute_process(COMMAND ${_spyre_ldd} "${TORCH_SPYRE_C_SO}"
                OUTPUT_VARIABLE _spyre_ldd_out ERROR_VARIABLE _spyre_ldd_err
                RESULT_VARIABLE _spyre_ldd_rc)
if(NOT _spyre_ldd_rc EQUAL 0)
  message(FATAL_ERROR "ldd ${TORCH_SPYRE_C_SO} failed:\n${_spyre_ldd_err}")
endif()
foreach(_pair "libflex.so|SPYRE_RUNTIME_DIR" "libspyre_comms.so.1|SPYRE_COMMS_DIR")
  string(REPLACE "|" ";" _pair "${_pair}")
  list(GET _pair 0 _lib)
  list(GET _pair 1 _var)
  string(REPLACE "." "\\." _lib_re "${_lib}")
  if(_spyre_ldd_out MATCHES "(^|\n)[ \t]*${_lib_re} => ([^ \n]+)")
    set(_found "${CMAKE_MATCH_2}")
  else()
    set(_found "")
  endif()
  if(NOT _found OR NOT EXISTS "${_found}")
    message(FATAL_ERROR
      "ldd ${TORCH_SPYRE_C_SO} does not resolve ${_lib}.\n"
      "LD_LIBRARY_PATH does not contain the Spyre stack this torch-spyre was built "
      "against. Fix LD_LIBRARY_PATH until `import torch_spyre._C` works, then "
      "re-derive the -D values with scripts/spyre_env.sh.")
  endif()
  get_filename_component(_found_real "${_found}" REALPATH)
  get_filename_component(_found_dir "${_found_real}" DIRECTORY)
  get_filename_component(_want_dir "${${_var}}/lib" REALPATH)
  message(STATUS "Spyre stack: ${TORCH_SPYRE_C_SO} loads ${_lib} from ${_found_dir}")
  if(NOT _found_dir STREQUAL _want_dir)
    message(FATAL_ERROR
      "Inconsistent Spyre stack: `ldd ${TORCH_SPYRE_C_SO}` resolves ${_lib} from\n"
      "    ${_found_dir}\n"
      "but ${_var}/lib is\n"
      "    ${_want_dir}\n"
      "LD_LIBRARY_PATH and the -D / environment values disagree. The runner would be "
      "linked against one stack and, through _C.so, load another (undefined symbols "
      "at link or load time). Make LD_LIBRARY_PATH list the stack you want first and "
      "re-derive the values with scripts/spyre_env.sh, or pass -D${_var} matching "
      "LD_LIBRARY_PATH. Reconfigure with `cmake --fresh` or in an EMPTY build "
      "directory: values in an old CMakeCache.txt are kept.")
  endif()
endforeach()

# RPATH for the runner executables (set in spyre_runner/CMakeLists.txt): the lib dirs
# just validated, the _C.so dir, and the dir of the libaiupti that ldd resolved (the
# runtime package ships it; sentient trees only carry its headers).
set(_spyre_rpath
    ${SPYRE_RUNTIME_DIR}/lib ${SPYRE_COMMS_DIR}/lib
    ${SPYRE_DEEPTOOLS_DIR}/lib ${SPYRE_SENLIB_DIR}/lib)
get_filename_component(_c_so_dir "${TORCH_SPYRE_C_SO}" DIRECTORY)
list(APPEND _spyre_rpath "${_c_so_dir}")
if(_spyre_ldd_out MATCHES "(^|\n)[ \t]*libaiupti\\.so => ([^ \n]+)")
  get_filename_component(_aiupti_real "${CMAKE_MATCH_2}" REALPATH)
  get_filename_component(_aiupti_dir "${_aiupti_real}" DIRECTORY)
  list(APPEND _spyre_rpath "${_aiupti_dir}")
endif()
list(REMOVE_DUPLICATES _spyre_rpath)
set(SPYRE_STACK_RPATH "${_spyre_rpath}" CACHE INTERNAL "RPATH dirs for the Spyre runners")

# nlohmann/json is header-only. Default to the copy vendored in the ExecuTorch tree
# (the headers only; the Spyre stack's own copy is an older version). Override with
# -DNLOHMANN_JSON_DIR=<dir containing include/nlohmann/json.hpp>.
set(NLOHMANN_JSON_DIR "${EXECUTORCH_ROOT}/third-party/json"
    CACHE PATH "nlohmann json dir (contains include/nlohmann/json.hpp)")
if(NOT EXISTS "${NLOHMANN_JSON_DIR}/include/nlohmann/json.hpp")
  message(FATAL_ERROR
    "nlohmann/json.hpp not found under NLOHMANN_JSON_DIR (${NLOHMANN_JSON_DIR}/include). "
    "Check EXECUTORCH_ROOT (${EXECUTORCH_ROOT}) or pass -DNLOHMANN_JSON_DIR=....")
endif()
message(STATUS "Spyre stack: NLOHMANN_JSON_DIR = ${NLOHMANN_JSON_DIR}")

add_library(spyre_backend STATIC
            ${CMAKE_CURRENT_SOURCE_DIR}/runtime/SpyreBackend.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/runtime/SpyreLayoutBridge.cpp)

target_compile_features(spyre_backend PUBLIC cxx_std_20)

# Mirror torch-spyre's own setup.py INCLUDE_DIRS so the transitive flex/senlib
# headers (messaging/, concurrentqueue, etc.) resolve exactly as they do there.
target_include_directories(
  spyre_backend
  PRIVATE ${NLOHMANN_JSON_DIR}/include
          ${TORCH_SPYRE_CSRC}
          ${SPYRE_RUNTIME_DIR}/include
          ${SPYRE_RUNTIME_DIR}/include/concurrentqueue/moodycamel
          ${SPYRE_SENLIB_DIR}/include
          ${SPYRE_COMMS_DIR}/include
          ${SPYRE_DEEPTOOLS_DIR}/include)

# torch-spyre headers include <torch/types.h> etc.; ${TORCH_INCLUDE_DIRS} and
# ${TORCH_LIBRARIES} come from find_package(Torch) in the parent scope. When
# building standalone, the driver script calls find_package(Torch) first.
target_include_directories(spyre_backend PRIVATE ${TORCH_INCLUDE_DIRS})

# prepare_kernel.h -> spyre_stream.h -> module.h pulls pybind11 -> <Python.h>,
# even though we only use the non-Python spyre:: entry points. Provide the
# interpreter headers so the include chain resolves.
find_package(Python3 COMPONENTS Development REQUIRED)
target_include_directories(spyre_backend PRIVATE ${Python3_INCLUDE_DIRS})

# ExecuTorch core (BackendInterface, register_backend, EValue, ...).
target_link_libraries(spyre_backend PUBLIC executorch_core)

# torch-spyre runtime symbols come straight from _C.so; the vendor stack libs
# resolve prepareKernel's transitive deps (flex / sendnn). _C.so lacks the
# "lib" prefix, so wrap it in an IMPORTED target: passing the bare path through
# target_link_libraries transitively gets mangled to "-l_C" by the linker line
# generator. An IMPORTED_LOCATION always links by absolute path.
add_library(torch_spyre_C SHARED IMPORTED GLOBAL)
set_target_properties(torch_spyre_C PROPERTIES
                      IMPORTED_LOCATION "${TORCH_SPYRE_C_SO}")

target_link_libraries(
  spyre_backend
  PUBLIC torch_spyre_C
         ${TORCH_LIBRARIES}
         Python3::Python
         # libflex is directly referenced by SpyreBackend.cpp (flex::CompositeAddress
         # etc.) and must be on the link line explicitly. The other Spyre runtime
         # symbols (senlib, deeptools, comms) resolve transitively through _C.so
         # and libflex's own NEEDED entries; libutil / libspyrecode-host-functions
         # are not referenced by this archive, so they are not linked directly.
         -L${SPYRE_RUNTIME_DIR}/lib
         -L${SPYRE_SENLIB_DIR}/lib
         -L${SPYRE_DEEPTOOLS_DIR}/lib
         -L${SPYRE_COMMS_DIR}/lib
         flex)

# The register_backend() global ctor must survive dead-code stripping when this
# static lib is linked into an executable. ExecuTorch ships this helper for
# exactly that (applies --whole-archive / -force_load around the target).
executorch_target_link_options_shared_lib(spyre_backend)

endif()  # EXECUTORCH_BUILD_SPYRE

# ---- Host-only unit tests ----------------------------------------------------
# The payload parser (runtime/SpyrePayload.h) is deliberately dependency-free, so
# its gtest links ONLY GoogleTest + nlohmann/json -- no torch, spyre, or
# executorch. That means it builds and runs on any host (CI, a laptop) with no
# Spyre device, mirroring the Python _pack_payload round-trip tests. Gated
# separately from EXECUTORCH_BUILD_SPYRE so CI can run it without the dev stack.
option(EXECUTORCH_BUILD_SPYRE_TESTS "Build the Spyre backend host-only unit tests" OFF)
if(EXECUTORCH_BUILD_SPYRE_TESTS)
  enable_testing()

  # Prefer a system/installed GoogleTest (e.g. Homebrew on the Mac, the distro
  # package in CI). When none is found -- as on the Spyre dev pod, which has no
  # system gtest -- fall back to building a GoogleTest source tree pointed to by
  # GOOGLETEST_SOURCE_DIR. Because it is a local path FetchContent uses it
  # directly (no network fetch), so the test build stays hermetic and offline.
  find_package(GTest QUIET)
  if(NOT TARGET GTest::gtest_main)
    set(GOOGLETEST_SOURCE_DIR ""
        CACHE PATH "GoogleTest source tree used when no system GTest is found")
    if(NOT EXISTS "${GOOGLETEST_SOURCE_DIR}/CMakeLists.txt")
      message(FATAL_ERROR
              "No system GTest found and GOOGLETEST_SOURCE_DIR "
              "('${GOOGLETEST_SOURCE_DIR}') is not a GoogleTest source tree. "
              "Install system GoogleTest (`brew install googletest` / "
              "`apt install libgtest-dev`) or pass "
              "-DGOOGLETEST_SOURCE_DIR=<path/to/googletest>.")
    endif()
    message(STATUS "Spyre tests: using vendored GoogleTest at "
                   "${GOOGLETEST_SOURCE_DIR}")
    include(FetchContent)
    FetchContent_Declare(googletest SOURCE_DIR "${GOOGLETEST_SOURCE_DIR}")
    # Don't install gtest or override the parent's compiler/runtime settings.
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
  endif()

  # nlohmann/json is header-only. Default to the copy vendored in the ExecuTorch
  # tree; override with a system/CI copy. Needs nothing from the Spyre stack.
  set(NLOHMANN_JSON_INCLUDE_DIR "${EXECUTORCH_ROOT}/third-party/json/include"
      CACHE PATH "Directory containing nlohmann/json.hpp for the host-only tests")

  add_executable(spyre_payload_test
                 ${CMAKE_CURRENT_SOURCE_DIR}/test/SpyrePayloadTest.cpp)
  target_compile_features(spyre_payload_test PRIVATE cxx_std_20)
  target_include_directories(spyre_payload_test
                             PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
                                     ${NLOHMANN_JSON_INCLUDE_DIR})
  target_link_libraries(spyre_payload_test PRIVATE GTest::gtest_main)
  include(GoogleTest)
  gtest_discover_tests(spyre_payload_test)

  # runtime/SpyreLayout.h (the device_layout schema) is likewise dependency-free.
  add_executable(spyre_layout_test
                 ${CMAKE_CURRENT_SOURCE_DIR}/test/SpyreLayoutTest.cpp)
  target_compile_features(spyre_layout_test PRIVATE cxx_std_20)
  target_include_directories(spyre_layout_test
                             PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
                                     ${NLOHMANN_JSON_INCLUDE_DIR})
  target_link_libraries(spyre_layout_test PRIVATE GTest::gtest_main)
  gtest_discover_tests(spyre_layout_test)
endif()
