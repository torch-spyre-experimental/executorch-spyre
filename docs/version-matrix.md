<!--
Copyright 2026 The ExecuTorch-Spyre Authors.
Licensed under the Apache License, Version 2.0.
-->

# Version matrix

This project requires a three-way version alignment between ExecuTorch, PyTorch,
and the torch-spyre SDK. Misalignment is a real risk: the wrong torch version can
silently break the C++ ABI between the ExecuTorch runtime and `torch_spyre/_C.so`.

## Pinned versions

| Component | Version | Notes |
|-----------|---------|-------|
| ExecuTorch | `v1.5.1` | Does not declare a torch pin, so `torch==2.13.0` is pinned explicitly. Install core only; skip `requirements-examples.txt`. |
| PyTorch | `2.13.0+cpu` | Must match torch-spyre's `_C.so` ABI. Used for both AOT export and the C++ device build. |
| Python | `3.12` | Used by CI and the device pod. The macOS arm64 wheel is built for macOS 14 or later. |
| torch-spyre SDK | `v0.5.0-rc.3` (2.0 image) | Submodule pinned to `v0.5.0-rc.3`. Use the prebuilt `_C.so` from the pod image. The SDK venv at `/home/senuser/torch-spyre/.venv` ships torch 2.13.0. |
| SDK image | IBM Spyre Software Stack, SDK 2.0 | A development build of the IBM Spyre Software Stack is required. Internal build instructions are available to IBM employees through internal documentation channels (same as torch-spyre). |
| C++ standard | C++20 | Required by `SpyreBackend.cpp`. Pass `-DCMAKE_CXX_STANDARD=20` if your toolchain does not default to it. |

## Known drift

### `install_executorch.sh` examples step pulls in extra packages

Running the examples step of `install_executorch.sh` installs
`requirements-examples.txt`, which pulls in `transformers`, `torchtune`, and related
packages that may pin incompatible versions of transitive dependencies.

**Fix:** Install ExecuTorch core only:

```bash
pip install "executorch==1.5.1" "torch==2.13.0+cpu"
```

### ExecuTorch v1.5.x drops the explicit torch dependency

ET 1.5.0 and 1.5.1 removed the explicit `torch` pin from their PyPI metadata
(the package now declares only `torchao` as a dependency). pip will not install
torch on its own, so `torch==2.13.0` must always be pinned alongside `executorch`.
The SDK venv already ships torch 2.13.0 so no separate install is needed in the pod.

## ExecuTorch release → torch pin reference

| ET release | PyTorch pin | Notes |
|------------|------------|-------|
| v1.2.0 | `torch==2.11.0` | Old SDK images. Do not use with the current 2.0 SDK image. |
| v1.3.1 | `torch==2.12.0` | ABI mismatch with SDK 2.0 `_C.so`. Do not use. |
| v1.4.1 | `torch>=2.13.0` | Aligns with SDK 2.0 image (torch-spyre `~=2.13.0`). Superseded. |
| **v1.5.1** ✅ (this project) | no explicit torch pin | Requires manual `torch==2.13.0+cpu` install alongside. Aligns with SDK 2.0 image. |

## See also

- [setup-reference.md](setup-reference.md) — host-only venv creation; [setup_pod.md](setup_pod.md) — pod environment setup.
- [debugging-guide.md](debugging-guide.md#pytorch-version-drift) — build
  troubleshooting for torch version drift.
