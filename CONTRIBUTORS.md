# Contributors

This repository began with a single squashed import commit
(`Initial import of ExecuTorch-Spyre v0.3.0rc1`), so `git log` does not show the
authorship of the work that preceded it. This file records that earlier work.

## Pre-import contributors

- **Michihiro Horie** (IBM) — original author of much of the backend and runtime, including:
  - the initial ExecuTorch backend for Spyre and its updates to the torch-spyre API;
  - Hugging Face Granite support: prefill, KV-cache decode path, the HFA export and runtime workflow,
    and integration with the ExecuTorch LLM runner;
  - the torch-spyre `fallback_ops`-driven partitioner capability source;
  - the pre-scheduling pass framework, `CompileSpec` pass selection and `chunk:N` partition granularity;
  - the build and CI configuration, the Spyre renaming, and the documentation restructure.

Commits made after the import are attributed in `git log`. To be added or to correct an entry,
open a pull request as described in [`CONTRIBUTING.md`](CONTRIBUTING.md).
