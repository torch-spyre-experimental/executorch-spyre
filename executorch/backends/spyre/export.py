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
"""Generic CLI: export any model to a Spyre-delegated ``.pte``.

    python -m executorch.backends.spyre.export --model <SPEC> [options]
    executorch-spyre-export --model <SPEC> [options]      # after `pip install -e .`

The per-model exporters (``export_matmul`` / ``export_mlp`` / ``export_mixed`` /
``export_attention``) each hard-code one module and
its inputs. This one takes the model as a spec and runs the same pipeline::

    torch.export -> to_edge -> .to_backend(SpyrePartitioner()) -> .to_executorch()

``--model`` accepts three kinds of spec (``--source auto`` picks by shape):

  * ``registry`` -- a name from upstream ExecuTorch's ``MODEL_NAME_TO_MODEL``
    catalogue (``linear``, ``softmax``, ``mv2``, ``resnet50``, ...). Some entries
    need extra packages (torchvision, transformers, ...) or downloaded weights.
  * ``module``   -- ``pkg.module:Attr`` or ``path/to/file.py:Attr``, where
    ``Attr`` is an ``nn.Module`` class, a factory returning one, an instance,
    or an upstream-style ``EagerModelBase``. Example inputs come from the
    model's ``get_example_inputs()`` or from ``--input``.
  * ``hf``       -- anything else: a Hugging Face model id or a local HF
    checkpoint dir, loaded with ``AutoModelForCausalLM`` (``AutoModel`` if it is
    not a causal LM), traced on random ``input_ids [batch, seq]`` with no KV
    cache, returning the logits (or the last hidden state).

Floating-point weights and inputs are cast to ``--dtype`` (fp16 by default,
the delegate's validated dtype). Whatever the partitioner does not delegate
falls back to the portable CPU kernels, and is reported.

Artifacts in ``--out-dir`` use the layout of the generic manifest-driven
``spyre_coverage_runner`` (see ``spyre_runner/spyre_coverage_runner.cpp``):

  * ``spyre_<name>.pte``
  * ``input<i>.bin`` -- each user input, raw little-endian, forward-arg order
  * ``ref.bin``      -- CPU reference for output 0 (computed in fp32, cast back)
  * ``manifest.txt`` -- the runner manifest
  * ``<name>_inputs.json`` -- the same, plus the delegated / CPU-fallback op sets

``--inspect`` stops after partitioning: it is device-free and needs no pod.
The default ``fallbacklist`` partition mode reads torch-spyre's ``fallback_ops``
and fails closed without ``torch_spyre``; on a host without it run with
``EXECUTORCH_SPYRE_PARTITION_MODE=allowlist``. Building the ``.pte`` runs ``SpyreBackend.preprocess``
(the SDSC compile) and so needs the Spyre toolchain; running it needs
``FLEX_DEVICE=PF`` and a runner linked with ``portable_ops_lib``.
"""

import argparse
import importlib
import importlib.util
import json
import logging
import os
import re
import sys
from pathlib import Path

import torch
from executorch.exir import EdgeCompileConfig, to_edge
from torch.utils._pytree import tree_flatten

from executorch.backends.spyre import SpyrePartitioner
from executorch.backends.spyre.export_attention import _partition_report

SOURCES = ("auto", "registry", "module", "hf")

# Manifest dtype token (spyre_coverage_runner ``parse_dtype``) <-> torch dtype.
_DTYPES: dict[str, torch.dtype] = {
    "fp16": torch.float16,
    "fp32": torch.float32,
    "int64": torch.int64,
    "bool": torch.bool,
}
_MANIFEST_DTYPE: dict[torch.dtype, str] = {v: k for k, v in _DTYPES.items()}

# Keep aten.silu fused through to_edge so its fp16->fp32 opmath decomposition
# does not insert CPU-falling-back casts.
_DEFAULT_PRESERVE_OPS = "aten.silu.default"

_INPUT_RE = re.compile(r"^(\d+(?:[,x]\d+)*)(?::(fp16|fp32|int64|bool))?$")


# --------------------------------------------------------------------------- #
# Spec parsing
# --------------------------------------------------------------------------- #


def resolve_source(spec: str, source: str = "auto") -> str:
    """Pick the loader for ``spec``: ``module`` / ``registry`` / ``hf``."""
    if source != "auto":
        return source
    if ":" in spec:  # HF ids and catalogue names never contain ':'
        return "module"
    if spec in _registry():
        return "registry"
    return "hf"


def parse_input_spec(text: str) -> tuple[list[int], "torch.dtype | None"]:
    """``"1,64"`` / ``"1x64:fp16"`` / ``"16:int64"`` -> ``(shape, dtype|None)``."""
    m = _INPUT_RE.match(text.strip())
    if not m:
        raise argparse.ArgumentTypeError(
            f"bad --input {text!r}: expected SHAPE[:DTYPE], e.g. 1,64 or 1x16:int64 "
            f"(DTYPE one of {', '.join(_DTYPES)})"
        )
    shape = [int(d) for d in re.split("[,x]", m.group(1))]
    return shape, (_DTYPES[m.group(2)] if m.group(2) else None)


def _parse_json_object(text: str) -> dict:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"--model-kwargs is not valid JSON: {e}")
    if not isinstance(value, dict):
        raise argparse.ArgumentTypeError("--model-kwargs must be a JSON object")
    return value


def _resolve_ops(text: str) -> list:
    """``"aten.silu.default,..."`` -> OpOverloads; ``"none"``/``""`` -> ``[]``."""
    if text.strip().lower() in ("", "none"):
        return []
    ops = []
    for name in text.split(","):
        obj = torch.ops
        try:
            for part in name.strip().split("."):
                obj = getattr(obj, part)
        except AttributeError:
            raise SystemExit(f"--preserve-ops: unknown op {name.strip()!r}")
        ops.append(obj)
    return ops


def _snake(name: str) -> str:
    """``"MixedUnsupported"`` / ``"ibm-granite/granite-3.3-8b"`` -> artifact-safe name."""
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
    return re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_").lower() or "model"


# --------------------------------------------------------------------------- #
# Model loaders: each returns (name, module, example_inputs | None)
# --------------------------------------------------------------------------- #


def _registry() -> dict:
    from executorch.examples.models import MODEL_NAME_TO_MODEL

    return MODEL_NAME_TO_MODEL


def _from_eager_model_base(obj) -> tuple[torch.nn.Module, tuple]:
    """Unpack upstream's ``EagerModelBase`` protocol (see ``EagerModelFactory``)."""
    get_kwargs = getattr(obj, "get_example_kwarg_inputs", None)
    if get_kwargs is not None and get_kwargs():
        raise SystemExit(
            "models with keyword example inputs are not supported: the device "
            "runner binds positional inputs only"
        )
    return obj.get_eager_model(), tuple(obj.get_example_inputs())


def _load_registry(spec: str, model_kwargs: dict):
    registry = _registry()
    if spec not in registry:
        raise SystemExit(
            f"{spec!r} is not in upstream MODEL_NAME_TO_MODEL; known: "
            + ", ".join(sorted(registry))
        )
    module_name, class_name = registry[spec]
    # Import directly rather than via EagerModelFactory, whose package prefix
    # depends on whether the cwd's basename is "executorch".
    try:
        pkg = importlib.import_module(f"executorch.examples.models.{module_name}")
    except ImportError as e:
        raise SystemExit(f"catalogue model {spec!r} needs a missing dependency: {e}")
    module, inputs = _from_eager_model_base(getattr(pkg, class_name)(**model_kwargs))
    return _snake(spec), module, inputs


def _import_attr(spec: str):
    target, _, attr = spec.rpartition(":")
    if not target or not attr:
        raise SystemExit(
            f"bad module spec {spec!r}: expected pkg.module:Attr or file.py:Attr"
        )
    if target.endswith(".py") or os.sep in target:
        path = Path(target).resolve()
        if not path.is_file():
            raise SystemExit(f"no such file: {path}")
        mod_name = f"_spyre_export_user_{_snake(path.stem)}"
        file_spec = importlib.util.spec_from_file_location(mod_name, path)
        module = importlib.util.module_from_spec(file_spec)
        sys.modules[mod_name] = module
        file_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(target)
    try:
        return attr, getattr(module, attr)
    except AttributeError:
        raise SystemExit(f"{target} has no attribute {attr!r}")


def _load_module(spec: str, model_kwargs: dict):
    attr, obj = _import_attr(spec)
    if isinstance(obj, torch.nn.Module):
        if model_kwargs:
            raise SystemExit(
                f"{spec} is already an nn.Module instance; drop --model-kwargs"
            )
    else:
        obj = obj(
            **model_kwargs
        )  # an nn.Module class, EagerModelBase class, or factory
    if hasattr(obj, "get_eager_model"):
        module, inputs = _from_eager_model_base(obj)
    elif isinstance(obj, torch.nn.Module):
        module = obj
        get_inputs = getattr(obj, "get_example_inputs", None)
        inputs = tuple(get_inputs()) if get_inputs is not None else None
    else:
        raise SystemExit(
            f"{spec} did not produce an nn.Module (got {type(obj).__name__})"
        )
    return _snake(attr), module, inputs


class _HFFirstOutput(torch.nn.Module):
    """``forward(input_ids) -> outputs[0]`` -- logits for a causal LM, else the
    last hidden state. Tuple outputs (``return_dict=False``) keep the exported
    graph's output a single tensor; the KV cache is off, so one forward is a
    full-sequence prefill."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids, return_dict=False)[0]


def _load_hf(spec: str, args, dtype: torch.dtype):
    try:
        import transformers
    except ImportError:
        raise SystemExit(
            f"{spec!r} looks like a Hugging Face model; `pip install transformers`"
        )
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    # Load straight into the target dtype (an 8B model in fp32 is 32 GB);
    # transformers 4.56 renamed `torch_dtype` to `dtype`.
    dtype_kw = "dtype" if (major, minor) >= (4, 56) else "torch_dtype"
    kwargs = {
        dtype_kw: dtype,
        # Plain-math attention: the SDPA kernel path is opaque to the partitioner.
        "attn_implementation": "eager",
        "trust_remote_code": args.trust_remote_code,
    }
    try:
        model = transformers.AutoModelForCausalLM.from_pretrained(spec, **kwargs)
    except ValueError:  # not a causal-LM architecture
        model = transformers.AutoModel.from_pretrained(spec, **kwargs)
    model.config.use_cache = False
    vocab = getattr(model.config, "vocab_size", None)
    if not vocab:
        raise SystemExit(
            f"{spec}: config has no vocab_size; only text models are supported"
        )
    input_ids = torch.randint(0, vocab, (args.batch, args.seq), dtype=torch.int64)
    return _snake(spec.rstrip("/").split("/")[-1]), _HFFirstOutput(model), (input_ids,)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _random_tensor(shape: list[int], dtype: torch.dtype) -> torch.Tensor:
    if dtype.is_floating_point:
        return torch.randn(shape).to(dtype)
    # Integer / bool inputs are 0/1: always-valid indices. Models that need
    # meaningful ids should provide get_example_inputs().
    return torch.randint(0, 2, shape).to(dtype)


def _cpu_reference(
    module: torch.nn.Module, inputs: tuple, dtype: torch.dtype
) -> torch.Tensor:
    """Output 0, computed in fp32 and cast back to ``dtype`` (the project's
    reference convention: CPU fp16 linear accumulates in reduced precision).
    The module is upcast in place and restored; fp16 -> fp32 -> fp16 is exact."""
    module.float()
    try:
        with torch.no_grad():
            out = module(*(t.float() if t.is_floating_point() else t for t in inputs))
    finally:
        module.to(dtype)
    leaves = [t for t in tree_flatten(out)[0] if isinstance(t, torch.Tensor)]
    if not leaves:
        raise SystemExit("model returned no tensor output")
    ref = leaves[0]
    return ref.to(dtype) if ref.is_floating_point() else ref


def _dump(path: str, t: torch.Tensor) -> None:
    """Write a tensor as raw little-endian bytes in its own dtype."""
    t.detach().contiguous().cpu().numpy().tofile(path)


def _manifest_line(kind: str, file: str, t: torch.Tensor) -> str:
    dims = " ".join(str(d) for d in t.shape)
    return f"{kind} {file} {_MANIFEST_DTYPE[t.dtype]} {t.dim()} {dims}".rstrip()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    # Show the name the user typed: the console script, or the `-m` form.
    console = os.path.basename(sys.argv[0]) == "executorch-spyre-export"
    ap = argparse.ArgumentParser(
        prog=(
            "executorch-spyre-export"
            if console
            else "python -m executorch.backends.spyre.export"
        ),
        description="Export a model to a Spyre-delegated .pte "
        "(torch.export -> to_edge -> SpyrePartitioner -> to_executorch).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="examples:\n"
        "  %(prog)s --model linear --inspect\n"
        "  %(prog)s --model executorch.backends.spyre.export_mixed:MixedUnsupported --input 1,64\n"
        "  %(prog)s --model ./my_model.py:Net --model-kwargs '{\"dim\": 64}' --input 1,64\n"
        "  %(prog)s --model ibm-granite/granite-3.3-2b-instruct --seq 128 --out-dir artifacts/",
    )
    ap.add_argument(
        "--model",
        required=True,
        help="upstream catalogue name, pkg.module:Attr / file.py:Attr, or a "
        "Hugging Face model id / local checkpoint dir",
    )
    ap.add_argument(
        "--source",
        choices=SOURCES,
        default="auto",
        help="how to interpret --model (default: auto -- ':' => module, a "
        "MODEL_NAME_TO_MODEL key => registry, else hf)",
    )
    ap.add_argument("--out-dir", default="artifacts")
    ap.add_argument("--name", help="artifact basename (default: derived from --model)")
    ap.add_argument(
        "--dtype",
        choices=("fp16", "fp32"),
        default="fp16",
        help="cast floating-point weights and inputs to this dtype (default: fp16)",
    )
    ap.add_argument(
        "--input",
        action="append",
        type=parse_input_spec,
        metavar="SHAPE[:DTYPE]",
        help="one per forward arg, in order, e.g. --input 1,64 --input 1x16:int64; "
        "random data. DTYPE defaults to --dtype. Overrides the model's own example "
        "inputs; required for a module without get_example_inputs()",
    )
    ap.add_argument(
        "--model-kwargs",
        type=_parse_json_object,
        default={},
        help="JSON object passed to the model class / factory constructor",
    )
    ap.add_argument(
        "--seq", type=int, default=128, help="hf: input_ids sequence length"
    )
    ap.add_argument("--batch", type=int, default=1, help="hf: input_ids batch size")
    ap.add_argument(
        "--trust-remote-code", action="store_true", help="hf: allow custom model code"
    )
    ap.add_argument(
        "--preserve-ops",
        default=_DEFAULT_PRESERVE_OPS,
        help="comma-separated ops kept un-decomposed through to_edge, or 'none' "
        f"(default: {_DEFAULT_PRESERVE_OPS})",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--inspect",
        action="store_true",
        help="device-free: trace + partition, print delegated / CPU-fallback ops, "
        "exit without building a .pte (no torch_spyre needed)",
    )
    ap.add_argument(
        "--no-reference",
        action="store_true",
        help="skip the CPU reference (ref.bin; manifest has no output line). "
        "The reference runs the model in fp32, which needs 2x its fp16 memory",
    )
    ap.add_argument(
        "-v", "--verbose", action="store_true", help="log every CPU-fallback node"
    )
    return ap


def _load(args, dtype: torch.dtype):
    source = resolve_source(args.model, args.source)
    if source == "registry":
        name, module, inputs = _load_registry(args.model, args.model_kwargs)
    elif source == "module":
        name, module, inputs = _load_module(args.model, args.model_kwargs)
    else:
        if args.model_kwargs:
            raise SystemExit("--model-kwargs is not supported for Hugging Face models")
        name, module, inputs = _load_hf(args.model, args, dtype)
    return source, name, module, inputs


def main(argv: "list[str] | None" = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    torch.manual_seed(args.seed)
    dtype = _DTYPES[args.dtype]
    preserve_ops = _resolve_ops(args.preserve_ops)

    source, name, module, inputs = _load(args, dtype)
    name = args.name or name
    module = module.eval().to(dtype)

    if args.input:
        inputs = tuple(_random_tensor(shape, d or dtype) for shape, d in args.input)
    elif inputs is None:
        raise SystemExit(
            f"{args.model} has no get_example_inputs(); pass --input SHAPE[:DTYPE]"
        )
    else:
        inputs = tuple(
            t.to(dtype) if isinstance(t, torch.Tensor) and t.is_floating_point() else t
            for t in inputs
        )
    if not all(isinstance(t, torch.Tensor) for t in inputs):
        raise SystemExit(
            "only tensor forward args are supported (the device runner binds tensors)"
        )

    print(f"model: {args.model}  (source={source}, dtype={args.dtype})")
    print(
        "inputs: "
        + ", ".join(
            f"{list(t.shape)}:{_MANIFEST_DTYPE.get(t.dtype, t.dtype)}" for t in inputs
        )
    )

    ep = torch.export.export(module, inputs)
    edge = to_edge(ep, compile_config=EdgeCompileConfig(preserve_ops=preserve_ops))

    delegated, fallback = _partition_report(edge.exported_program())
    print(f"delegated ops ({len(delegated)} nodes): {sorted(set(delegated)) or 'none'}")
    print(
        f"CPU-fallback ops ({len(fallback)} nodes): {sorted(set(fallback)) or 'none'}"
    )

    # A preserved op is by construction not core ATen, and the portable CPU
    # kernels implement core ATen only -- so one left on CPU would fail at run
    # time with OperatorMissing (0x14). Refuse to build such a .pte.
    stuck = sorted({op._schema.name for op in preserve_ops} & set(fallback))
    if stuck:
        raise SystemExit(
            f"preserved op(s) {stuck} were not delegated and have no portable CPU "
            "kernel; re-run with --preserve-ops none to let them decompose"
        )
    if args.inspect:
        return

    ref = None if args.no_reference else _cpu_reference(module, inputs, dtype)

    et = edge.to_backend(SpyrePartitioner()).to_executorch()

    os.makedirs(args.out_dir, exist_ok=True)
    pte_name = f"spyre_{name}.pte"
    pte_path = os.path.join(args.out_dir, pte_name)
    with open(pte_path, "wb") as f:
        f.write(et.buffer)

    meta_inputs = []
    for i, t in enumerate(inputs):
        file = f"input{i}.bin"
        _dump(os.path.join(args.out_dir, file), t)
        meta_inputs.append(
            {
                "file": file,
                "dtype": _MANIFEST_DTYPE.get(t.dtype, str(t.dtype)),
                "shape": list(t.shape),
            }
        )
    meta_output = None
    if ref is not None:
        _dump(os.path.join(args.out_dir, "ref.bin"), ref)
        meta_output = {
            "file": "ref.bin",
            "dtype": _MANIFEST_DTYPE.get(ref.dtype, str(ref.dtype)),
            "shape": list(ref.shape),
        }

    # spyre_coverage_runner manifest; it understands only the _DTYPES tokens.
    tensors = list(inputs) + ([ref] if ref is not None else [])
    if all(t.dtype in _MANIFEST_DTYPE for t in tensors):
        manifest = [f"pte {pte_name}"]
        manifest += [
            _manifest_line("input", f"input{i}.bin", t) for i, t in enumerate(inputs)
        ]
        if ref is not None:
            manifest.append(_manifest_line("output", "ref.bin", ref))
        with open(os.path.join(args.out_dir, "manifest.txt"), "w") as f:
            f.write("\n".join(manifest) + "\n")
    else:
        print(
            f"note: no manifest.txt -- the runner supports only {', '.join(_DTYPES)} tensors"
        )

    meta = {
        "pte": pte_name,
        "model": args.model,
        "source": source,
        "dtype": args.dtype,
        "inputs": meta_inputs,
        "output": meta_output,
        "n_outputs": len(ep.graph_signature.user_outputs),
        "delegated_ops": sorted(set(delegated)),
        "cpu_fallback_ops": sorted(set(fallback)),
    }
    with open(os.path.join(args.out_dir, f"{name}_inputs.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"wrote {pte_path} ({len(et.buffer)} bytes)")


if __name__ == "__main__":
    main()
