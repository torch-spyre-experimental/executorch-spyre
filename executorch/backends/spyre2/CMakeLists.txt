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
# spyre2 runtime build — reuse spyre's device backend wholesale, add the spyre2 alias.
#
# The C++ runtime is codegen-independent and identical to spyre's: the spyre2 Stage 2
# (SDSC) payload is byte-compatible with spyre's, and the parser/tensor-bridge are
# reused from executorch/backends/spyre/. The ONLY runtime difference is that a spyre2 `.pte`
# carries backend_id "Spyre2Backend" (so ExecuTorch's AOT resolver, which matches
# backend_id == the Python class __name__, routes to Spyre2Backend.preprocess
# instead of spyre's). The runtime must therefore register the same stateless
# BackendInterface under that name too.
#
# Rather than duplicate spyre's Spyre-stack build config, we add_subdirectory() the
# spyre backend and flip on SPYRE_REGISTER_SPYRE2_ALIAS, which activates the guarded
# second register_backend("Spyre2Backend", ...) in SpyreBackend.cpp. spyre's own build
# never defines this macro, so its device-verified binary is byte-identical.

cmake_minimum_required(VERSION 3.19)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(spyre2_backend LANGUAGES C CXX)
endif()

# Root of the ExecuTorch source tree. Default assumes the standard open-source
# layout: executorch/ is a sibling of this repo's checkout.
set(EXECUTORCH_ROOT
    "${CMAKE_CURRENT_SOURCE_DIR}/../executorch"
    CACHE PATH "Root of the ExecuTorch source tree")

# spyre backend source dir (holds the real CMake + the shared SpyreBackend.cpp).
# In the open-source repo this is always executorch/backends/spyre/ in this same repo.
set(SPYRE_BACKEND_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../spyre"
    CACHE PATH "spyre backend source dir (reused device runtime + build config)")

# Optional: override to a pre-built GoogleTest tree; leave empty to let
# spyre's CMake fetch or locate it.
set(GOOGLETEST_SOURCE_DIR
    ""
    CACHE PATH "Pre-built GoogleTest source tree (leave empty for auto-detect)")

if(NOT EXISTS "${SPYRE_BACKEND_DIR}/CMakeLists.txt")
  message(FATAL_ERROR
          "SPYRE_BACKEND_DIR ('${SPYRE_BACKEND_DIR}') is not the spyre backend "
          "source tree. Point -DSPYRE_BACKEND_DIR at executorch/backends/spyre.")
endif()

# Build spyre's spyre_backend target (device lib) in a subdir of our binary tree.
add_subdirectory(${SPYRE_BACKEND_DIR} spyre_backend_build)

# Register the runtime under spyre2's backend id as well. PRIVATE: only this
# compilation of SpyreBackend.cpp gets the alias; the source stays shared.
if(TARGET spyre_backend)
  target_compile_definitions(spyre_backend PRIVATE SPYRE_REGISTER_SPYRE2_ALIAS)
  message(STATUS "spyre2: spyre_backend will also register the \"Spyre2Backend\" id.")
endif()
