<!--
Copyright 2026 The ExecuTorch-Spyre Authors.
Licensed under the Apache License, Version 2.0.
-->

# Contributing to the Spyre ExecuTorch backend

This backend is developed to **upstream quality** so it can land in either
`pytorch/executorch` (`executorch/backends/spyre`) or a torch-spyre `executorch` subpackage.
Follow the process both projects require.

Issues should be filed in this repo's [GitHub issue tracker](../../issues).

## Developer Certificate of Origin (DCO)

Both target projects gate merges on a DCO check. **Every commit must be signed
off**:

```bash
git commit -s -m "executorch/backends/spyre: <what changed>"
```

`-s` appends a `Signed-off-by: Your Name <you@example.com>` trailer certifying
the [DCO](https://developercertificate.org/). Amend an existing commit with
`git commit --amend -s`; fix a whole branch with
`git rebase --signoff <base>`. CI rejects any commit lacking the trailer, so
configure `git config user.name` / `user.email` to your real identity first.

## Style & linting

- **C++** — Google C++ style via the [.clang-format](.clang-format) in this
  directory (`BasedOnStyle: Google`, C++20). This matches both ExecuTorch and
  torch-spyre.
- **Python** — `black` + `ruff` (see [.pre-commit-config.yaml](.pre-commit-config.yaml)),
  Google Python docstring conventions.

Install the hooks once and run them before every push:

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

## Tests must pass

The host-only suites run with no Spyre device and are the CI gate; the
device/end-to-end suite is a separate manual job on a Spyre dev pod. Exact
commands are in [docs/setup-reference.md](docs/setup-reference.md#host-only-check-no-device). At minimum, before
opening a PR:

```bash
# Python AOT unit tests (partitioner + preprocess round-trip)
PYTHONPATH="$PWD:$PYTHONPATH" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  pytest -p no:cacheprovider executorch/backends/spyre/test/

# C++ payload parser gtest (uses system GoogleTest if present; otherwise falls
# back to ExecuTorch's vendored googletest via -DGOOGLETEST_SOURCE_DIR — see
# docs/setup-reference.md)
cmake -S executorch/backends/spyre -B /tmp/spyre-hosttest-build \
  -DEXECUTORCH_BUILD_SPYRE=OFF -DEXECUTORCH_BUILD_SPYRE_TESTS=ON
cmake --build /tmp/spyre-hosttest-build
ctest --test-dir /tmp/spyre-hosttest-build --output-on-failure
```

New op coverage in `_SUPPORTED_ATEN_OPS` should come with a partitioner test and,
where feasible, an on-device end-to-end check within the fp16 envelope
(`atol=1e-1, rtol=2e-2`).

## PR process

1. **Open an issue first** for anything beyond a small fix — especially the large
   initial contribution — in this repo's [GitHub issue tracker](../../issues).
2. Keep the C++ (`runtime/`) and Python (AOT) sides consistent: the payload wire
   format is shared, so a change to `_pack_payload` needs a matching change to
   `SpyrePayload.h` **and** both round-trip tests.
3. Sign off, lint-clean, tests green — then submit.
