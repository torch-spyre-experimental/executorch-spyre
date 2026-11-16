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
# Standalone driver for the Phase 2 exit test. Builds the Spyre backend delegate
# and a tiny runner that executes a Spyre-delegated .pte on device and compares
# against a CPU fp16 reference. NOT part of the upstream backend layout — kept
# under backend/spyre/spyre_runner/ so the whole spike travels together.
#
# Configure (in-pod, from a fresh, EMPTY build dir), pointing at an ExecuTorch
# INSTALL tree (cmake --install) and the AOT venv's torch. The Spyre stack
# (TORCH_SPYRE_C_SO, TORCH_SPYRE_CSRC, SPYRE_{RUNTIME,DEEPTOOLS,SENLIB,COMMS}_DIR) has
# no defaults; derive it from the _C.so that Python imports:
#
#   mapfile -t SPYRE_CMAKE_ARGS < <(scripts/spyre_env.sh --cmake-args)
#   cmake -S executorch/backends/spyre/spyre_runner -B build-spyre \
#     -DCMAKE_PREFIX_PATH="$ET_INSTALL;$(python -c 'import torch,os;print(os.path.dirname(torch.__file__))')/share/cmake" \
#     -Dexecutorch_DIR="$ET_INSTALL/lib64/cmake/ExecuTorch" \
#     -DEXECUTORCH_ROOT="$ET_SRC" \
#     "${SPYRE_CMAKE_ARGS[@]}"
#   cmake --build build-spyre -j
#   scripts/spyre_env.sh --check build-spyre/spyre_llm_runner
#
# Expected ExecuTorch install layout (note lib64, and the mixed-case
# `ExecuTorch` directory; the config file itself is lowercase):
#
#   <install>/lib64/cmake/ExecuTorch/executorch-config.cmake   -> executorch_DIR
#   <install>/lib64/cmake/tokenizers/tokenizers-config.cmake   -> tokenizers_DIR
#                                                  (LLM-runner installs only)
#   <install>/lib64/lib*.a                                     (static libs)
#
# $ET_SRC is the ExecuTorch SOURCE tree (provides tools/cmake/Utils.cmake); it
# should be the same ExecuTorch version that produced <install>. `lib64` is the
# GNUInstallDirs default on RHEL-family hosts; other hosts may use `lib`.
#
# Do NOT set executorch_DIR to a path derived from an unset variable: an empty
# prefix silently becomes "/lib/cmake/ExecuTorch" and CMake then reports only
# "Could not find a package configuration file provided by executorch".
#
# Some ExecuTorch target names differ across install layouts (extension_module
# vs extension_module_static, presence of extension_data_loader). The runner
# link list below is patched at first build against what the tree actually
# exports — see documentation for details.

cmake_minimum_required(VERSION 3.19)
project(spyre_runner CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)
find_package(executorch CONFIG REQUIRED)

# Build the delegate (defines target `spyre_backend`).
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/.. spyre_backend_build)

# spyre_backend is whole-archive-linked (see ../CMakeLists.txt) so its static
# register_backend() ctor is retained. The ET extensions give Module +
# from_blob; ${TORCH_LIBRARIES} provides c10::Half for the fp16 compare.
# All runners share the identical link list; only the .pte + I/O differ.
# spyre_coverage_runner is the generic, manifest-driven runner for the Phase 5
# op-coverage device gate (coverage/run_coverage.py::_gate_device).
#
# portable_ops_lib registers ExecuTorch's portable CPU kernels. Without it the
# runner can execute delegated (Spyre) ops but ANY node left on CPU by the
# partitioner fails at run time with 0x14 (OperatorMissing). The SDSC context
# matmul (softmax-fed `mm`) is a justified CPU fallback — see documentation —
# so the runner must carry the portable kernels for that fallback to execute.
# The imported target already supplies `--whole-archive` link options, so
# listing it here retains the static op-registration ctors automatically.
#
# A FULLY delegated graph (e.g. the T4 single-op `mm`: the whole forward is one
# executorch_call_delegate, no CPU node) needs none of this. Turning the option
# OFF then also lets ExecuTorch itself be built without the portable-kernels
# codegen (gen_oplist), which requires the `torchgen.executorch` overlay absent
# from the SDK 2.0 image's torch. Default ON preserves the Phase 8 device build.
option(SPYRE_RUNNER_WITH_PORTABLE_OPS
       "Link ExecuTorch portable CPU kernels into the runners (needed for CPU-fallback nodes)" ON)

set(_spyre_runner_libs
    spyre_backend
    executorch_core
    extension_module_static
    extension_tensor
    extension_data_loader)
if(SPYRE_RUNNER_WITH_PORTABLE_OPS)
  list(APPEND _spyre_runner_libs portable_ops_lib)
endif()
list(APPEND _spyre_runner_libs ${TORCH_LIBRARIES})

# BUILD_RPATH: the lib dirs of the Spyre stack the runner was configured against
# (SPYRE_*_DIR, validated against `ldd _C.so` in ../CMakeLists.txt), the _C.so dir,
# and the dir of the libaiupti that `ldd _C.so` resolves. With
# this the runner finds libflex / libspyre_comms / DeepTools without LD_LIBRARY_PATH
# and an /opt/ibm entry on LD_LIBRARY_PATH can no longer substitute another libflex.
# It is a DT_RPATH (--disable-new-dtags), which the loader searches BEFORE
# LD_LIBRARY_PATH and also for the transitive dependencies of these libraries.
# Python is different: the embedded interpreter's `import torch_spyre` loads _C.so,
# which has no RPATH of its own, so PYTHONPATH / LD_LIBRARY_PATH still matter for the
# exports and for _C.so's own dependencies (see docs/setup-reference.md).
# SPYRE_STACK_RPATH is computed and validated in ../CMakeLists.txt.
set(_spyre_rpath "${SPYRE_STACK_RPATH}")
message(STATUS "Spyre stack: runner BUILD_RPATH = ${_spyre_rpath}")

foreach(runner spyre_runner spyre_mlp_runner spyre_coverage_runner)
  add_executable(${runner} ${CMAKE_CURRENT_SOURCE_DIR}/${runner}.cpp)
  target_include_directories(${runner} PRIVATE ${TORCH_INCLUDE_DIRS})
  target_link_libraries(${runner} PRIVATE ${_spyre_runner_libs})
  set_target_properties(${runner} PROPERTIES BUILD_RPATH "${_spyre_rpath}")
  target_link_options(${runner} PRIVATE -Wl,--disable-new-dtags)
endforeach()

# spyre_llm_runner: drives an --llm-wrapper PTE through the STANDARD ExecuTorch LLM
# runner stack (TextDecoderRunner / TextPrefiller / TextTokenGenerator /
# TextLLMRunner / create_text_llm_runner). Opt-in: it needs an ExecuTorch install
# built with EXECUTORCH_BUILD_EXTENSION_LLM_RUNNER=ON (extension_llm_runner +
# tokenizers), which the default install does not provide.
option(SPYRE_BUILD_LLM_RUNNER
       "Build spyre_llm_runner (requires an ExecuTorch installation built with EXECUTORCH_BUILD_EXTENSION_LLM_RUNNER=ON)" OFF)
if(SPYRE_BUILD_LLM_RUNNER)
  # extension_llm_runner / extension_llm_sampler are imported targets created
  # by find_package(executorch) above, only if the install provides them.
  # tokenizers is a separate package (<install>/lib64/cmake/tokenizers).
  # Probe quietly, then fail here with an actionable message instead of a
  # distant "package not found" / unresolved-target error.
  find_package(tokenizers CONFIG QUIET)
  set(_spyre_llm_missing "")
  foreach(_t extension_llm_runner extension_llm_sampler tokenizers::tokenizers)
    if(NOT TARGET ${_t})
      list(APPEND _spyre_llm_missing ${_t})
    endif()
  endforeach()
  if(_spyre_llm_missing)
    string(REPLACE ";" ", " _spyre_llm_missing_str "${_spyre_llm_missing}")
    message(FATAL_ERROR
      "SPYRE_BUILD_LLM_RUNNER=ON but the ExecuTorch installation in use does not "
      "provide: ${_spyre_llm_missing_str}\n"
      "  executorch_DIR = ${executorch_DIR}\n"
      "  tokenizers_DIR = ${tokenizers_DIR}\n"
      "This requires an ExecuTorch install configured with\n"
      "  -DEXECUTORCH_BUILD_EXTENSION_LLM=ON\n"
      "  -DEXECUTORCH_BUILD_EXTENSION_LLM_RUNNER=ON\n"
      "  -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON\n"
      "and installed with `cmake --install`. Point -Dexecutorch_DIR at "
      "<install>/lib64/cmake/ExecuTorch (and make sure <install> is on "
      "CMAKE_PREFIX_PATH so tokenizers is found), or configure without "
      "-DSPYRE_BUILD_LLM_RUNNER=ON.")
  endif()
  message(STATUS "spyre_llm_runner: enabled (executorch_DIR=${executorch_DIR})")
  add_executable(spyre_llm_runner ${CMAKE_CURRENT_SOURCE_DIR}/spyre_llm_runner.cpp)
  target_include_directories(spyre_llm_runner PRIVATE ${TORCH_INCLUDE_DIRS})
  target_link_libraries(spyre_llm_runner PRIVATE
      ${_spyre_runner_libs}
      extension_llm_runner
      extension_llm_sampler
      tokenizers::tokenizers)
  set_target_properties(spyre_llm_runner PROPERTIES BUILD_RPATH "${_spyre_rpath}")
  target_link_options(spyre_llm_runner PRIVATE -Wl,--disable-new-dtags)
else()
  message(STATUS
    "spyre_llm_runner: disabled (pass -DSPYRE_BUILD_LLM_RUNNER=ON; needs an "
    "ExecuTorch install with EXECUTORCH_BUILD_EXTENSION_LLM_RUNNER=ON)")
endif()
