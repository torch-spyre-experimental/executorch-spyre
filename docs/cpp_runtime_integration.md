# Integrating ExecuTorch-Spyre Runtime in C++ Applications

This guide explains how C++ applications embed and interact with the Spyre runtime to execute models delegated to IBM Spyre hardware.

```mermaid
flowchart TD
    subgraph BuildStep["1. Build & Link Integration"]
        UserApp["Your C++ Application (e.g. main.cpp)"]
        LibSpyreBackend["libspyre_backend.a"]
        ETLibs["ExecuTorch Core & Extensions"]
        SpyreLibs["torch-spyre (_C.so) & libflex"]

        UserApp --> LinkStage["Linker with --whole-archive / -force_load"]
        LibSpyreBackend --> LinkStage
        ETLibs --> LinkStage
        SpyreLibs --> LinkStage
        LinkStage --> FinalBin["Application Binary"]
    end

    subgraph RuntimeStep["2. Runtime Execution Flow"]
        FinalBin -->|1. Static Init| Reg["register_backend('SpyreBackend')"]
        FinalBin -->|2. Load Model| ModInit["Module module('model.pte')"]
        ModInit -->|Calls delegate init()| Init["SpyreBackend::init()<br/>- Unpacks kernel bundles<br/>- Calls spyre::prepareKernel()"]
        FinalBin -->|3. Forward Pass| Forward["module.forward(inputs)"]
        Forward -->|Calls delegate execute()| Exec["SpyreBackend::execute()<br/>- Copies inputs Host -> Spyre<br/>- Launches Spyre JobPlans<br/>- Copies outputs Spyre -> Host"]
    end
```

---

## 1. Architecture Overview

Connecting an external C++ application to the ExecuTorch-Spyre backend relies on three primary mechanisms:

1. **Static Delegate Registration**:
   `SpyreBackend.cpp` defines a static `SpyreBackend` instance and registers it via ExecuTorch's `register_backend({"SpyreBackend", &g_spyre_backend})` during global static initialization.
2. **Whole-Archive Linking**:
   The static library `spyre_backend` must be linked with `--whole-archive` (Linux) or `-force_load` (macOS) so the linker does not prune the static registration constructor during dead-code elimination.
3. **ExecuTorch Module API**:
   The application uses `executorch::extension::Module` to load `.pte` files and execute `module.forward(...)`. When ExecuTorch encounters a delegated partition with `backend_id == "SpyreBackend"`, it automatically routes initialization to `SpyreBackend::init()` and execution to `SpyreBackend::execute()`.

---

## 2. Standard C++ Application Example

Below is a complete template showing how a C++ application loads a Spyre-delegated `.pte` program, wraps host memory buffers, runs inference, and retrieves results:

```cpp
#include <iostream>
#include <vector>
#include <memory>

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>

using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::extension::from_blob;
using executorch::extension::Module;
using executorch::extension::TensorPtr;
using executorch::runtime::Error;
using executorch::runtime::EValue;

int run_inference(const std::string& model_path,
                  void* input_data_ptr,
                  const std::vector<int32_t>& input_sizes,
                  ScalarType dtype) {
  // 1. Wrap host input memory as an ExecuTorch TensorPtr (zero-copy wrapper)
  TensorPtr input_tensor = from_blob(input_data_ptr, input_sizes, dtype);

  // 2. Load the delegated .pte program.
  //    ExecuTorch resolves "SpyreBackend" from the model metadata and invokes
  //    SpyreBackend::init() to extract kernel bundles and prepare Spyre JobPlans.
  Module module(model_path);
  if (!module.is_loaded()) {
    std::cerr << "Failed to load model: " << model_path << std::endl;
    return -1;
  }

  // 3. Execute inference.
  //    ExecuTorch invokes SpyreBackend::execute(), transferring inputs to the
  //    Spyre device, launching compiled JobPlans, and copying outputs back.
  auto result = module.forward({EValue(*input_tensor)});
  if (result.error() != Error::Ok) {
    std::cerr << "Inference failed with error code: 0x"
              << std::hex << static_cast<unsigned>(result.error()) << std::endl;
    return -1;
  }

  // 4. Retrieve outputs
  Tensor out_tensor = result->at(0).toTensor();
  const void* out_data = out_tensor.const_data_ptr();
  size_t num_elements = out_tensor.numel();

  std::cout << "Inference completed successfully. Output elements: "
            << num_elements << std::endl;

  return 0;
}
```

---

## 3. CMake Build Configuration

When compiling your application, link against `spyre_backend` along with required ExecuTorch and PyTorch/Spyre dependencies:

```cmake
cmake_minimum_required(VERSION 3.19)
project(my_spyre_app CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)
find_package(executorch CONFIG REQUIRED)

# Add the Spyre backend target (exports spyre_backend with whole-archive options)
add_subdirectory(path/to/executorch-spyre/executorch/backends/spyre spyre_backend_build)

add_executable(my_spyre_app main.cpp)

target_include_directories(my_spyre_app PRIVATE ${TORCH_INCLUDE_DIRS})

target_link_libraries(my_spyre_app PRIVATE
    spyre_backend               # Spyre Delegate runtime with static registration
    portable_ops_lib          # Portable CPU kernels for non-delegated operations
    executorch_core
    extension_module_static
    extension_tensor
    extension_data_loader
    ${TORCH_LIBRARIES}
)
```

---

## 4. Key Runtime Considerations

| Feature | Details & Best Practices |
|---|---|
| **CPU Fallback Operators** | If any model operators are not partitioned to Spyre, `portable_ops_lib` must be linked into the binary. Omitting it will result in runtime error `0x14` (`OperatorMissing`). |
| **Dynamic Shapes & Bucketing** | Output buffers in `.pte` are planned for the upper-bound bucket shape. `SpyreBackend::execute()` writes the active bucket's result to the front of the allocated memory; downstream code should slice tensors according to the active batch/sequence dimensions. |
| **Diagnostics & Profiling** | Set `EXECUTORCH_SPYRE_PROFILE=1` in your environment to enable log breakdowns for H2D copies, kernel launches, device synchronization, and D2H copies. |
| **Device Execution Mode** | Ensure `FLEX_DEVICE=PF` is set in the runtime environment when executing on physical Spyre cards. |
