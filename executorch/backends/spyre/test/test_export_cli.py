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
"""Host (device-free) tests for the generic export CLI (``spyre/export.py``).

The SDSC compile inside ``SpyreBackend.preprocess`` needs the Spyre toolchain,
so the end-to-end tests stub it with a fixed payload: everything else -- model
loading for each ``--model`` source, ``torch.export`` / ``to_edge`` /
``SpyrePartitioner`` / ``to_executorch``, the ``.pte`` serialization and the
runner artifacts -- is the real pipeline. Device execution of the result is the
generic ``spyre_coverage_runner`` (reads the emitted ``manifest.txt``).
"""

import json
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("executorch")

from executorch.exir._serialize._program import deserialize_pte_binary  # noqa: E402
from executorch.exir.backend.backend_details import PreprocessResult  # noqa: E402

from executorch.backends.spyre import export as cli  # noqa: E402
from executorch.backends.spyre.spyre_backend import SpyreBackend  # noqa: E402

_STUB = b"spyre-stub-payload"

_MIXED = "executorch.backends.spyre.export_mixed:MixedUnsupported"


@pytest.fixture
def stub_preprocess(monkeypatch):
    monkeypatch.setattr(
        SpyreBackend,
        "preprocess",
        staticmethod(
            lambda edge_program, compile_specs: PreprocessResult(processed_bytes=_STUB)
        ),
    )


def _delegate_ids(pte_path) -> list[str]:
    with open(pte_path, "rb") as f:
        program = deserialize_pte_binary(f.read()).program
    return [d.id for plan in program.execution_plan for d in plan.delegates]


def _read_manifest(out_dir) -> list[str]:
    return (out_dir / "manifest.txt").read_text().splitlines()


# --- argument parsing -----------------------------------------------------


def test_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["--help"])
    assert e.value.code == 0
    assert "--model" in capsys.readouterr().out


def test_module_main_help():
    # `python -m executorch.backends.spyre.export --help` in a fresh interpreter:
    # needs the package importable there (`pip install -e .` or conftest-like
    # path setup), so skip rather than fail on a bare PYTHONPATH checkout.
    proc = subprocess.run(
        [sys.executable, "-m", "executorch.backends.spyre.export", "--help"],
        capture_output=True,
        text=True,
    )
    if "No module named 'executorch.backends.spyre'" in proc.stderr:
        pytest.skip("executorch.backends.spyre not installed (pip install -e .)")
    assert proc.returncode == 0, proc.stderr
    assert "--model" in proc.stdout


@pytest.mark.parametrize(
    "spec, expected",
    [
        ("linear", "registry"),
        (_MIXED, "module"),
        ("./my_model.py:Net", "module"),
        ("ibm-granite/granite-3.3-2b-instruct", "hf"),
        ("/models/granite", "hf"),
    ],
)
def test_resolve_source(spec, expected):
    assert cli.resolve_source(spec) == expected


def test_resolve_source_explicit_wins():
    assert cli.resolve_source("linear", "hf") == "hf"


@pytest.mark.parametrize(
    "text, shape, dtype",
    [
        ("1,64", [1, 64], None),
        ("1x64", [1, 64], None),
        ("16:int64", [16], torch.int64),
        ("2,3,4:fp32", [2, 3, 4], torch.float32),
        ("1,8:bool", [1, 8], torch.bool),
    ],
)
def test_parse_input_spec(text, shape, dtype):
    assert cli.parse_input_spec(text) == (shape, dtype)


@pytest.mark.parametrize("text", ["", "1,", "a,b", "1,64:int8", "1,64:"])
def test_parse_input_spec_rejects(text):
    with pytest.raises(Exception):
        cli.parse_input_spec(text)


def test_module_without_example_inputs_needs_input(tmp_path):
    with pytest.raises(SystemExit, match="--input"):
        cli.main(["--model", "torch.nn:ReLU", "--inspect", "--out-dir", str(tmp_path)])


# --- inspect (partition only, no .pte) -------------------------------------


def test_inspect_writes_nothing(tmp_path, capsys, monkeypatch):
    # `sin` is a CPU fallback only in allowlist mode; under the default
    # fallbacklist mode torch-spyre delegates it.
    monkeypatch.setenv("EXECUTORCH_SPYRE_PARTITION_MODE", "allowlist")
    out = tmp_path / "out"
    cli.main(["--model", _MIXED, "--input", "1,64", "--inspect", "--out-dir", str(out)])
    text = capsys.readouterr().out
    assert "CPU-fallback ops (1 nodes): ['aten::sin']" in text
    assert not out.exists()


def test_preserved_op_on_cpu_is_refused(tmp_path):
    # A last dim > 64 at the graph output is refused by the partitioner
    # (_writes_scrambled_output), so the preserved aten.silu would land on CPU,
    # where portable has no silu kernel -> OperatorMissing at run time.
    with pytest.raises(SystemExit, match="--preserve-ops none"):
        cli.main(
            [
                "--model",
                "torch.nn:SiLU",
                "--input",
                "1,128",
                "--inspect",
                "--out-dir",
                str(tmp_path),
            ]
        )
    # Letting it decompose instead is accepted.
    cli.main(
        [
            "--model",
            "torch.nn:SiLU",
            "--input",
            "1,128",
            "--inspect",
            "--preserve-ops",
            "none",
        ]
    )


# --- full export (stubbed SDSC compile) ------------------------------------


def test_export_module_class_path(tmp_path, stub_preprocess, monkeypatch):
    # Allowlist mode keeps `sin` on CPU (see test_inspect_writes_nothing).
    monkeypatch.setenv("EXECUTORCH_SPYRE_PARTITION_MODE", "allowlist")
    cli.main(["--model", _MIXED, "--input", "1,64", "--out-dir", str(tmp_path)])

    pte = tmp_path / "spyre_mixed_unsupported.pte"
    assert pte.is_file()
    # The interior CPU `sin` splits the graph into two SpyreBackend partitions.
    assert _delegate_ids(pte) == ["SpyreBackend", "SpyreBackend"]

    assert _read_manifest(tmp_path) == [
        "pte spyre_mixed_unsupported.pte",
        "input input0.bin fp16 2 1 64",
        "output ref.bin fp16 2 1 64",
    ]
    assert (tmp_path / "input0.bin").stat().st_size == 64 * 2
    assert (tmp_path / "ref.bin").stat().st_size == 64 * 2

    meta = json.loads((tmp_path / "mixed_unsupported_inputs.json").read_text())
    assert meta["source"] == "module"
    assert meta["cpu_fallback_ops"] == ["aten::sin"]
    assert "aten::addmm" in meta["delegated_ops"]


def test_export_reference_matches_fp32_eager(tmp_path, stub_preprocess):
    from executorch.backends.spyre.export_mixed import MixedUnsupported

    cli.main(
        [
            "--model",
            _MIXED,
            "--input",
            "1,64",
            "--seed",
            "3",
            "--out-dir",
            str(tmp_path),
        ]
    )

    # Same seed => same weights, then same input draw.
    torch.manual_seed(3)
    model = MixedUnsupported().eval().to(torch.float16)
    x = torch.randn(1, 64).to(torch.float16)
    expected = model.float()(x.float()).to(torch.float16)

    import numpy as np

    got = torch.from_numpy(
        np.fromfile(tmp_path / "input0.bin", dtype=np.float16)
    ).reshape(1, 64)
    ref = torch.from_numpy(np.fromfile(tmp_path / "ref.bin", dtype=np.float16)).reshape(
        1, 64
    )
    assert torch.equal(got, x)
    assert torch.equal(ref, expected)


def test_export_registry_model(tmp_path, stub_preprocess):
    cli.main(["--model", "linear", "--out-dir", str(tmp_path)])

    pte = tmp_path / "spyre_linear.pte"
    assert _delegate_ids(pte) == ["SpyreBackend"]
    # LinearModule.get_example_inputs() -> one [3, 3] input, cast to fp16.
    assert _read_manifest(tmp_path) == [
        "pte spyre_linear.pte",
        "input input0.bin fp16 2 3 3",
        "output ref.bin fp16 2 3 3",
    ]
    meta = json.loads((tmp_path / "linear_inputs.json").read_text())
    assert meta["source"] == "registry"
    assert meta["cpu_fallback_ops"] == []


def test_export_file_path_with_kwargs(tmp_path, stub_preprocess):
    src = tmp_path / "my_model.py"
    src.write_text(
        "import torch\n"
        "class Net(torch.nn.Module):\n"
        "    def __init__(self, dim):\n"
        "        super().__init__()\n"
        "        self.fc = torch.nn.Linear(dim, dim)\n"
        "    def forward(self, x):\n"
        "        return torch.relu(self.fc(x))\n"
        "    def get_example_inputs(self):\n"
        "        return (torch.randn(1, self.fc.in_features),)\n"
    )
    out = tmp_path / "out"
    cli.main(
        [
            "--model",
            f"{src}:Net",
            "--model-kwargs",
            '{"dim": 32}',
            "--name",
            "net",
            "--out-dir",
            str(out),
        ]
    )
    assert _delegate_ids(out / "spyre_net.pte") == ["SpyreBackend"]
    assert _read_manifest(out)[1] == "input input0.bin fp16 2 1 32"


def test_export_no_reference(tmp_path, stub_preprocess):
    cli.main(["--model", "add", "--no-reference", "--out-dir", str(tmp_path)])
    assert not (tmp_path / "ref.bin").exists()
    assert _read_manifest(tmp_path) == [
        "pte spyre_add.pte",
        "input input0.bin fp16 1 1",
        "input input1.bin fp16 1 1",
    ]


def test_export_hf_checkpoint(tmp_path, stub_preprocess):
    transformers = pytest.importorskip("transformers")
    # A tiny random-weight Granite saved as a local checkpoint: exercises the
    # same from_pretrained path as a hub id, with no network access.
    config = transformers.GraniteConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        max_position_embeddings=64,
    )
    ckpt = tmp_path / "tiny-granite"
    transformers.GraniteForCausalLM(config).save_pretrained(ckpt)

    out = tmp_path / "out"
    cli.main(["--model", str(ckpt), "--seq", "8", "--out-dir", str(out)])

    assert "SpyreBackend" in _delegate_ids(out / "spyre_tiny_granite.pte")
    assert _read_manifest(out) == [
        "pte spyre_tiny_granite.pte",
        "input input0.bin int64 2 1 8",
        "output ref.bin fp16 3 1 8 64",
    ]
    meta = json.loads((out / "tiny_granite_inputs.json").read_text())
    assert meta["source"] == "hf"
    # The preserved silu must have been delegated (else main() refuses).
    assert "aten::silu" not in meta["cpu_fallback_ops"]
