# Copyright 2026 The ExecuTorch-Spyre Authors.
# Licensed under the Apache License, Version 2.0.
"""Functional-validation export: run hf-adapters' HFA execution path.

Generic replacement for the former ``export_granite_hfa.py``.  Works with any
standard-GQA family supported by hf-adapters (Granite, Llama, Qwen2, Mistral,
Ministral) — adapter selection is driven entirely by the checkpoint's
``config.json`` (AutoConfig → hf_adapters.auto_spyre_model dispatch).

For synthetic models (no ``--ckpt``) the script synthesises a small Granite
model, which has the same execution path as every other standard-GQA family.

Unlike a hand-written re-implementation of the decoder, this script exports
``hf_<family>._run_forward`` unchanged:

    checkpoint (save_pretrained/from_pretrained, fp16)
      -> adapter.prepare_for_spyre(model)      # torch.compile -> identity only
      -> adapter._run_forward(...)             # embed_text_tokens, _spyre_rope,
                                               # StandardGQABlock x N, norm, lm_head
      -> torch.export -> to_edge -> Spyre partition -> .pte
      -> spyre_coverage_runner, compared to the *HF fp32* logits

Every divergence from hf-adapters is behind an explicit ``--rope`` / ``--kv`` /
``--norm`` / ``--sdpa`` switch so each can be proven necessary (or not).

Goal is functional execution, NOT delegation quality: fallbacks are reported
but never gate anything.
"""

import argparse
import copy
import os
import sys
import types

import torch
import torch.nn as nn

hf_adapters_root = os.environ.get("HF_ADAPTERS_ROOT")
if not hf_adapters_root:
    raise RuntimeError("HF_ADAPTERS_ROOT must point to the hf-adapters repository")
sys.path.insert(0, hf_adapters_root)
_pkg = types.ModuleType(
    "hf_adapters"
)  # skip hf_adapters/__init__ (pulls every adapter)
_pkg.__path__ = [os.path.join(hf_adapters_root, "hf_adapters")]
_pkg.__package__ = "hf_adapters"
sys.modules.setdefault("hf_adapters", _pkg)

from hf_adapters import hf_common  # noqa: E402

hf_common.DEVICE = (
    "cpu"  # same patch hf-adapters' tests/conftest.py applies for CPU tracing
)
from hf_adapters.hf_common import (  # noqa: E402
    PrecomputedRotaryEmbedding,
    build_decode_mask,
    build_prefill_mask,
    generation_cache_len,
    get_backbone,
    text_config,
    make_cache_index,
    prebuild_rope_cache,
    set_rope_dtype,
    untie_embedding_and_lm_head,
)

from executorch.exir import EdgeCompileConfig, to_edge  # noqa: E402

from executorch.backends.spyre import SpyrePartitioner  # noqa: E402
from executorch.backends.spyre.spyre_block_partition import (  # noqa: E402
    format_partition_report,
)
from executorch.backends.spyre.export_attention import (  # noqa: E402
    _dump_fp16,
    _partition_report,
)


# ---------------------------------------------------------------------------
# Adapter lookup (config-driven, no --family flag required)
# ---------------------------------------------------------------------------


def _get_adapter_for_config(cfg):
    """Return the hf_adapters module appropriate for a config object.

    Dispatch mirrors hf_adapters.auto_spyre_model.CONFIG_TO_ADAPTER_MODULE_MAPPING
    but is evaluated lazily here so we don't import every adapter at startup.
    Only the standard-GQA families in scope are listed; others fall back to
    hf_granite (the well-tested baseline and default for synthetic models).

    This variant accepts a config object directly and is usable *before* the
    model is loaded (e.g. to select a custom loader).
    """
    from transformers import (
        LlamaConfig,
        MistralConfig,
        Qwen2Config,
    )

    try:
        from transformers.models.ministral.configuration_ministral import (
            MinistralConfig,
        )
    except ImportError:
        MinistralConfig = None
    try:
        from transformers.models.mistral3.configuration_mistral3 import Mistral3Config
    except ImportError:
        Mistral3Config = None

    if isinstance(cfg, LlamaConfig):
        from hf_adapters import hf_llama

        return hf_llama
    if isinstance(cfg, MistralConfig):
        from hf_adapters import hf_mistral

        return hf_mistral
    if MinistralConfig is not None and isinstance(cfg, MinistralConfig):
        from hf_adapters import hf_ministral

        return hf_ministral
    if Mistral3Config is not None and isinstance(cfg, Mistral3Config):
        from hf_adapters import hf_mistral3

        return hf_mistral3
    if isinstance(cfg, Qwen2Config):
        from hf_adapters import hf_qwen2

        return hf_qwen2
    # Default: Granite (covers GraniteConfig and synthetic models)
    from hf_adapters import hf_granite

    return hf_granite


def truncate_backbone_layers(model, n_layers):
    """Keep only the first *n_layers* decoder layers of an already-loaded *model*.

    Custom adapter loaders (e.g. hf_mistral3.load_hf_model) re-read config.json from
    disk, so an in-memory ``num_hidden_layers`` override never reaches the model.
    This truncates the loaded backbone itself and keeps the loaded model's own config
    in sync (hf_common derives KV shapes from ``text_config(model.config)``).
    A no-op when *n_layers* is falsy or already equals the loaded layer count.
    """
    if not n_layers:
        return
    backbone = get_backbone(model)
    total = len(backbone.layers)
    if n_layers > total:
        raise ValueError(
            f"--num-layers {n_layers} exceeds the {total} layers in the loaded model"
        )
    if n_layers == total:
        return
    backbone.layers = backbone.layers[:n_layers]
    tcfg = getattr(model.config, "text_config", model.config)
    tcfg.num_hidden_layers = n_layers
    layer_types = getattr(tcfg, "layer_types", None)
    if layer_types is not None:
        tcfg.layer_types = list(layer_types)[:n_layers]


def check_layer_consistency(model, n_layers):
    """Fail fast if the prepared model's block count disagrees with *n_layers*
    (the count used to allocate KV caches)."""
    n_blocks = len(model._spyre_compiled_blocks)
    if n_blocks != n_layers:
        raise RuntimeError(
            f"layer/cache mismatch after prepare_for_spyre: "
            f"len(model._spyre_compiled_blocks)={n_blocks} but args.layers={n_layers} "
            f"(KV caches are allocated per args.layers; backbone iterates every compiled block)"
        )


def _get_adapter(model):
    """Return the hf_adapters module appropriate for *model* (post-load).

    Thin wrapper around _get_adapter_for_config — kept for callers that already
    have a loaded model (HfaExport, post-load adapter resolve).
    """
    return _get_adapter_for_config(model.config)


def stage(name, ok, detail=""):
    print(f"STAGE {name}: {'PASS' if ok else 'FAIL'} {detail}", flush=True)


# ---------------------------------------------------------------------------
# Switchable divergences from hf-adapters
# ---------------------------------------------------------------------------
class GatherRope(PrecomputedRotaryEmbedding):
    """PrecomputedRotaryEmbedding.forward minus the host-side .item()/_extend_cache.

    Same cache, same indexing; just no data-dependent python control flow.
    (Cache must be prebuilt with prebuild_rope_cache.)
    """

    def forward(self, hidden_states, position_ids):
        return self._freq_cache[position_ids]


class HfaExport(nn.Module):
    """torch.export container around adapter._run_forward (unchanged).

    ``kv_mode="input"`` / ``return_kv=True`` are for the CPU reference only (no CLI route exports them).
    """

    def __init__(
        self,
        model,
        adapter=None,
        n_layers=None,
        kv_shape=None,
        kv_mode="input",
        return_kv=False,
        dtype=torch.float16,
    ):
        super().__init__()
        if isinstance(adapter, int):
            # Legacy signature: HfaExport(model, n_layers, kv_shape, kv_mode, return_kv=False)
            n_layers, kv_shape, kv_mode = adapter, n_layers, kv_shape
            adapter = _get_adapter(model)
        elif adapter is None:
            adapter = _get_adapter(model)
        self.model = model
        self.adapter = adapter
        self.kv_mode = kv_mode
        self.n = n_layers
        # return_kv is only meaningful with kv_mode="input": buffer mode cannot return
        # updated buffers as graph outputs because torch.export treats registered buffers
        # as parameters (constants), not data-flow values.
        self.return_kv = return_kv and (kv_mode == "input")
        if kv_mode == "buffer":
            for i in range(n_layers):
                # dtype must match the model's weight dtype: KV caches are consumed by
                # SDPA as key/value alongside q_proj output — a dtype mismatch produces
                # fp32 upcasts that the Spyre SDSC kernel rejects.  Default is fp16 so
                # callers that do not pass dtype= are unaffected.
                self.register_buffer(f"kc{i}", torch.zeros(kv_shape, dtype=dtype))
                self.register_buffer(f"vc{i}", torch.zeros(kv_shape, dtype=dtype))

    def forward(self, input_ids, position_ids, attn_mask, cache_index, *caches):
        if self.kv_mode == "buffer":
            kcs = [getattr(self, f"kc{i}") for i in range(self.n)]
            vcs = [getattr(self, f"vc{i}") for i in range(self.n)]
        else:
            kcs = list(caches[: self.n])
            vcs = list(caches[self.n :])
        m = self.model
        logits = self.adapter._run_forward(
            m, input_ids, position_ids, attn_mask, kcs, vcs, cache_index
        )
        # pad_lm_head pads the vocab to a tile-friendly width; hf-adapters crops back to
        # config.vocab_size (hf_common.py: next_logits[:, :vocab_size]).  A no-op when aligned.
        logits = logits[..., : text_config(m.config).vocab_size]
        if self.return_kv:
            # Return updated KV caches alongside logits so torch.export produces a
            # graph with 1 + 2*n_layers outputs: (logits, kc0, …, kcN-1, vc0, …, vcN-1).
            # kcs/vcs have been updated in-place by _run_forward via index_copy_; returning
            # them here makes the mutations visible as data-flow outputs in the graph.
            return (logits, *kcs, *vcs)
        return logits


# ---------------------------------------------------------------------------
# Standard ExecuTorch LLM runner interface wrapper
# ---------------------------------------------------------------------------


class StandardLLMWrapper(nn.Module):
    """Wraps HfaExport to expose the standard ExecuTorch LLM decode signature.

    Exported forward signature::

        forward(tokens: int64[1, 1], input_pos: int64[1]) -> logits: fp16[1, 1, V]

    The ``attn_mask``, ``position_ids`` and ``cache_index`` required by the
    hf-adapters path are all derived **inside the graph** so the caller only
    needs to supply:

      * ``tokens``      – the single next token id, shape ``[1, 1]`` int64.
      * ``input_pos``   – the current cache write position, shape ``[1]`` int64.

    Derivations (all graph-traceable, no host-side .item() calls):

      * ``position_ids = input_pos.view(1, 1)``   → ``[1, 1]`` int64
      * ``cache_index  = input_pos``               → ``[1]``   int64
      * ``attn_mask``  – ``[1, 1, 1, C]`` fp16/bf16: row ``input_pos`` of a
        constant ``[C, C]`` mask table (embedding gather).  Positions
        0..input_pos get 0.0; input_pos+1..C-1 get the fill value.

    KV caches are registered buffers (always ``kv_mode="buffer"``); the caller
    is responsible for loading prefill state into them before calling
    ``torch.export.export``.

    Parameters
    ----------
    model        : prepared HF model (output of adapter.prepare_for_spyre).
    adapter      : resolved hf_adapters module (e.g. hf_granite).
    n_layers     : number of decoder layers (== len(model._spyre_compiled_blocks)).
    kv_shape     : shape of each KV cache tensor: ``(1, n_kv, C, head_dim)``.
    cache_len    : C — the KV cache sequence length (= kv_shape[2]).
    mask_fill    : fill value for masked positions (default: fp16 half-min / 2).
    dtype        : model/KV dtype (torch.float16 or torch.bfloat16).
    """

    # fp16 fill value: finfo(float16).min / 2 = -32752.0.  This matches the
    # hf-adapters ``_mask_fill_value`` policy for BOTH fp16 and bf16 checkpoints
    # because Spyre stores both as SEN169_FP16 internally.
    _DEFAULT_MASK_FILL: float = -32752.0

    def __init__(
        self,
        model,
        adapter,
        n_layers: int,
        kv_shape,
        cache_len: int,
        mask_fill: float = _DEFAULT_MASK_FILL,
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self._inner = HfaExport(
            model,
            adapter,
            n_layers,
            kv_shape,
            kv_mode="buffer",
            return_kv=False,
            dtype=dtype,
        )
        self._C = cache_len
        self._dtype = dtype
        # Precomputed decode-mask table [C, C]: row p is the [C] additive mask for
        # input_pos == p (0 for slots <= p, mask_fill beyond).  The in-graph mask
        # is a single embedding gather by input_pos.
        #
        # The earlier formulation (arange > input_pos[0] -> where, with expand of
        # the scalar fill) partitions `gt`/`expand_copy` onto Spyre.  It compiles
        # but the delegated mask sub-graph cannot be launched on device
        # ("Number of inputs provided (2) does not match number of inputs
        # expected (3)"); fp16 arithmetic/compare variants of the mask run but
        # produce wrong values.  The gather stays on CPU (same placement as the
        # old `where`) and is bit-exact.
        _pos = torch.arange(cache_len)
        _tab = torch.zeros(cache_len, cache_len, dtype=dtype)
        _tab[_pos[None, :] > _pos[:, None]] = mask_fill
        self.register_buffer("_mask_table", _tab)

    def forward(self, tokens: torch.Tensor, input_pos: torch.Tensor) -> torch.Tensor:
        """tokens: int64[1,1]   input_pos: int64[1]   -> logits: fp_[1,1,V]"""
        # --- derive inner arguments from the two inputs ---
        # position_ids: [1, 1] int64 (position of the single new token)
        position_ids = input_pos.view(1, 1)
        # cache_index: [1] int64  (same as input_pos — the slot to write)
        cache_index = input_pos
        # attn_mask: [1, 1, 1, C] = row `input_pos` of the constant mask table.
        attn_mask = torch.nn.functional.embedding(input_pos, self._mask_table).view(
            1, 1, 1, self._C
        )
        return self._inner(tokens, position_ids, attn_mask, cache_index)

    # ------------------------------------------------------------------
    # Delegate buffer access to the inner HfaExport so callers can still
    # do ``getattr(wrapper, "kc0")`` / ``copy_`` to inject prefill state.
    # ------------------------------------------------------------------
    def _kv_buffer_name(self, i: int, kind: str) -> str:
        return f"{kind}c{i}"

    def kc(self, i: int) -> torch.Tensor:
        return getattr(self._inner, f"kc{i}")

    def vc(self, i: int) -> torch.Tensor:
        return getattr(self._inner, f"vc{i}")


def _detect_eos_ids(ckpt: str, override: str = ""):
    """Return the checkpoint's EOS token ids as a de-duplicated ``list[int]``.

    Precedence: ``--eos-ids`` > ``generation_config.json`` > ``config.json``
    (``text_config`` for multimodal wrappers).  ``eos_token_id`` may be an int or
    a list.  Returns ``[]`` when nothing is declared (the runner then falls back
    to the tokenizer's single ``eos_tok()``).
    """
    import json

    if override:
        raw = [int(x) for x in override.split(",") if x.strip()]
    else:
        raw = []
        for fn, nested in (
            ("generation_config.json", None),
            ("config.json", "text_config"),
        ):
            fp = os.path.join(ckpt, fn)
            if not os.path.exists(fp):
                continue
            with open(fp) as f:
                d = json.load(f)
            if nested and isinstance(d.get(nested), dict):
                d = {**d, **d[nested]}
            v = d.get("eos_token_id")
            if v is not None:
                raw = [v] if isinstance(v, int) else list(v)
                break
    return list(dict.fromkeys(int(i) for i in raw))


def assert_eos_exported(eos_ids, constant_methods):
    """Fail export when several EOS ids exist but ``get_eos_ids`` is not exported.

    The stock ExecuTorch LLM runner (``llm::get_eos_ids``) only knows the
    tokenizer's single ``eos_tok()`` unless the PTE exports ``get_eos_ids``;
    with several EOS ids (e.g. end-of-text + end-of-turn) generation would not
    stop on the secondary ids and silently run to ``max_new_tokens``.
    """
    if len(eos_ids) > 1 and "get_eos_ids" not in (constant_methods or {}):
        raise SystemExit(
            f"ERROR: multiple EOS ids {eos_ids} detected but get_eos_ids is not exported; "
            f"refusing to export a PTE the LLM runner cannot stop correctly."
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--vocab", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--inter", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=1)
    ap.add_argument("--kv-heads", type=int, default=1)
    ap.add_argument("--head-dim", type=int, default=0, help="0 => hidden/heads")
    ap.add_argument("--seq", type=int, default=16)
    ap.add_argument("--cache-len", type=int, default=0, help="0 => seq")
    ap.add_argument(
        "--mults",
        type=float,
        default=1.0,
        help="use non-trivial Granite multipliers (value applied to all; synthetic model only)",
    )
    ap.add_argument(
        "--qk-scale",
        type=float,
        default=1.0,
        help="scale q/k/embedding weights so attention + RoPE are numerically significant",
    )
    ap.add_argument(
        "--ckpt",
        default="",
        help="existing HF checkpoint dir (overrides synthetic config)",
    )
    ap.add_argument("--rope", choices=["orig", "gather"], default="orig")
    ap.add_argument("--kv", choices=["buffer"], default="buffer")
    ap.add_argument("--norm", choices=["orig"], default="orig")
    ap.add_argument(
        "--mode",
        choices=["prefill", "decode"],
        default="prefill",
        help="prefill: seq=S query len; decode: query len=1 with cache populated from prefill",
    )
    ap.add_argument("--stage", choices=["export", "edge", "pte"], default="pte")
    ap.add_argument(
        "--partition-granularity",
        default=None,
        help="compile unit: block (default; 1 block/delegate) | "
        "model (monolithic: all blocks in 1 delegate) | "
        "chunk:<N> (N blocks/delegate). "
        "Overrides EXECUTORCH_SPYRE_PARTITION_GRANULARITY.",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--num-layers",
        type=int,
        default=0,
        help="with --ckpt: keep only the first N decoder layers (bring-up); 0 => all",
    )
    ap.add_argument(
        "--prompt",
        default="",
        help="with --ckpt: tokenize this text (padded/truncated to --seq) instead of random ids",
    )
    ap.add_argument(
        "--eos-ids",
        default="",
        help="with --llm-wrapper: comma-separated EOS ids overriding the checkpoint's "
        "generation_config.json / config.json eos_token_id. get_eos_ids is "
        "exported when more than one EOS id is in effect.",
    )
    ap.add_argument(
        "--llm-wrapper",
        action="store_true",
        help="export StandardLLMWrapper: forward(tokens[1,1], input_pos[1]) -> logits "
        "(standard ExecuTorch LLM runner interface; decode + --kv buffer only; "
        "adds constant_methods: get_max_seq_len, get_max_context_len, "
        "use_kv_cache=True, enable_dynamic_shape=False)",
    )
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    S = args.seq
    # For decode: cache must hold at least S+1 slots (slot S is the decode write position).
    # generation_cache_len rounds up to the nearest SDPA-compatible block boundary.
    C = args.cache_len or (generation_cache_len(S, 1) if args.mode == "decode" else S)

    # ---------------- checkpoint -> from_pretrained ----------------
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        GraniteConfig,
        GraniteForCausalLM,
    )

    m = args.mults
    cfg = GraniteConfig(
        vocab_size=args.vocab,
        hidden_size=args.hidden,
        intermediate_size=args.inter,
        num_hidden_layers=args.layers,
        num_attention_heads=args.heads,
        num_key_value_heads=args.kv_heads,
        max_position_embeddings=max(64, C),
        embedding_multiplier=m,
        residual_multiplier=m,
        attention_multiplier=(
            m
            if m != 1.0
            else 1.0 / ((args.head_dim or args.hidden // args.heads) ** 0.5)
        ),
        logits_scaling=m,
        tie_word_embeddings=True,
        use_cache=False,
    )
    if args.head_dim:
        cfg.head_dim = args.head_dim
    ckpt = os.path.join(args.out_dir, "ckpt")
    if args.ckpt:
        ckpt = args.ckpt
    else:
        raw = GraniteForCausalLM(cfg)
        if args.qk_scale != 1.0:
            with torch.no_grad():
                raw.model.embed_tokens.weight.mul_(args.qk_scale)
                for l in raw.model.layers:
                    l.self_attn.q_proj.weight.mul_(args.qk_scale)
                    l.self_attn.k_proj.weight.mul_(args.qk_scale)
        raw.save_pretrained(ckpt)
    if args.ckpt:
        # Real checkpoint: every dimension comes from ITS config, not the synthetic-CLI defaults.
        rcfg = AutoConfig.from_pretrained(ckpt)
        # Dimension read-out: multimodal configs (e.g. Mistral3Config) nest the
        # text-decoder dims under .text_config; plain configs expose them directly.
        _dcfg = getattr(rcfg, "text_config", rcfg)
        if args.num_layers:
            _dcfg.num_hidden_layers = args.num_layers
        # Resolve adapter early so we can use its custom loader when present.
        # Mistral3-family models (Mistral3Config) cannot be loaded via
        # AutoModelForCausalLM — they are multimodal checkpoints registered only
        # under AutoModelForImageTextToText. Their adapter (hf_mistral3) provides
        # load_hf_model() which uses Mistral3ForConditionalGeneration directly.
        _load_adapter = _get_adapter_for_config(rcfg)
        # Resolve dtype via the official policy (same as AutoSpyreModel.from_pretrained).
        # target_device="spyre" so fp32 configs are promoted to fp16, matching the device path.
        from executorch.backends.spyre._dtype_utils import dtype_for_model_path

        model_dtype = dtype_for_model_path(ckpt, target_device="spyre")
        if hasattr(_load_adapter, "load_hf_model"):
            model = _load_adapter.load_hf_model(ckpt, model_dtype)
            # load_hf_model re-reads config.json from disk, ignoring the --num-layers
            # override applied to rcfg above: truncate the loaded model explicitly.
            truncate_backbone_layers(model, args.num_layers)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                ckpt, config=rcfg, torch_dtype=model_dtype
            )
        args.layers = _dcfg.num_hidden_layers
        args.heads = _dcfg.num_attention_heads
        args.kv_heads = _dcfg.num_key_value_heads
        args.hidden = _dcfg.hidden_size
        args.vocab = _dcfg.vocab_size
        args.head_dim = (
            getattr(_dcfg, "head_dim", None)
            or _dcfg.hidden_size // _dcfg.num_attention_heads
        )
        C = args.cache_len or (
            generation_cache_len(S, 1) if args.mode == "decode" else S
        )
    else:
        # Synthetic checkpoint saved by this script: always fp16 (no policy table entry).
        model = GraniteForCausalLM.from_pretrained(ckpt, torch_dtype=torch.float16)
        model_dtype = torch.float16
    model = model.eval().requires_grad_(False)
    ref_model = (
        copy.deepcopy(model).float().eval()
    )  # HF reference, untouched by hf-adapters
    stage("load_checkpoint", True, f"{ckpt}")

    # Resolve adapter module (config-driven, no --family flag)
    adapter = _get_adapter(model)
    stage("resolve_adapter", True, f"adapter={adapter.__name__.split('.')[-1]}")

    # ---------------- hf-adapters preparation (verbatim) ----------------
    untie_embedding_and_lm_head(model)
    real_compile = torch.compile
    torch.compile = (
        lambda f, **kw: f
    )  # only change: let torch.export see through the wrappers
    try:
        adapter.prepare_for_spyre(model)
    finally:
        torch.compile = real_compile
    check_layer_consistency(model, args.layers)
    padded_hd = model._spyre_rope.padded_head_dim
    stage(
        "adapter.prepare_for_spyre",
        True,
        f"padded_head_dim={padded_hd} blocks={len(model._spyre_compiled_blocks)}",
    )

    # ---------------- divergence switches ----------------
    # Apply official dtype policy to the RoPE freq cache before building it.
    # In the AutoSpyreModel path this is done by _move_to_spyre_with_layout →
    # set_rope_dtype → prebuild_rope_cache.  We replicate the same two-step
    # sequence here so the cache is built in the correct dtype (avoids fp16/bf16
    # mismatch in apply_rope_matmul for bf16 checkpoints such as Ministral).
    set_rope_dtype(model, model_dtype)
    if args.rope == "gather":
        prebuild_rope_cache(model)
        r = model._spyre_rope
        g = GatherRope(r.original, r.padded_head_dim)
        g._freq_cache, g._cached_len = r._freq_cache, r._cached_len
        model._spyre_rope = g
    else:
        prebuild_rope_cache(
            model
        )  # same as hf-adapters' _move_to_spyre_with_layout prelude
        if args.llm_wrapper:
            # Under --llm-wrapper, position_ids is derived from the symbolic
            # input_pos tensor (input_pos.view(1,1)), so its value is unknown at
            # torch.export time.  PrecomputedRotaryEmbedding.forward does:
            #
            #   max_pos = int(pos_cpu.max().item()) + 1   # raises GuardOnDataDependentSymNode
            #   self._extend_cache(max_pos)
            #
            # The cache has already been fully pre-built by prebuild_rope_cache
            # above, so _extend_cache always early-returns in practice — the only
            # work that matters is the index: _freq_cache[position_ids].
            # Replace with GatherRope (same cache, no .item()) so torch.export
            # traces cleanly.  The rope dict case (Gemma/ModernBERT) is left
            # as-is: those families are not supported by --llm-wrapper.
            r = model._spyre_rope
            if not isinstance(r, dict):
                g = GatherRope(r.original, r.padded_head_dim)
                g._freq_cache, g._cached_len = r._freq_cache, r._cached_len
                model._spyre_rope = g

    # ---------------- example inputs (as hf-adapters generate() builds them) ----------------
    n_kv = args.kv_heads
    hd = padded_hd if padded_hd else (args.head_dim or args.hidden // args.heads)
    kv_shape = (1, n_kv, C, hd)

    # --- Build mode-specific inputs ---
    if args.mode == "prefill":
        input_ids = torch.randint(0, args.vocab, (1, S), dtype=torch.int64)
        if args.ckpt and args.prompt:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(ckpt)
            ids = tok(args.prompt, return_tensors="pt").input_ids[0][:S]
            input_ids[0, : ids.numel()] = ids
            if (
                ids.numel() < S
            ):  # pad by repeating the prompt so every position is a real token
                input_ids[0] = ids.repeat((S + ids.numel() - 1) // ids.numel())[:S]
        position_ids = torch.arange(S, dtype=torch.int64).unsqueeze(0)
        # attn_mask is added to Q@K^T inside SDPA; dtype must match model dtype.
        attn_mask = build_prefill_mask(1, S, C, 0, dtype=model_dtype)
        cache_index = make_cache_index(0, S, "cpu")
        caches = []
    else:  # decode
        # Build decode inputs: single token at cache position S (just after prefill).
        # KV caches are populated by running a CPU prefill first so the decode
        # eager-vs-HF comparison has realistic KV state.
        #
        # Step 1: run CPU prefill to populate KV caches and get the next token.
        prefill_ids = torch.randint(0, args.vocab, (1, S), dtype=torch.int64)
        pf_pos_ids = torch.arange(S, dtype=torch.int64).unsqueeze(0)
        pf_mask = build_prefill_mask(1, S, C, 0, dtype=model_dtype)
        pf_ci = make_cache_index(0, S, "cpu")
        pf_kcs = [torch.zeros(kv_shape, dtype=model_dtype) for _ in range(args.layers)]
        pf_vcs = [torch.zeros(kv_shape, dtype=model_dtype) for _ in range(args.layers)]
        with torch.no_grad():
            pf_wrapper = HfaExport(
                model,
                adapter,
                args.layers,
                kv_shape,
                "input",
                return_kv=True,
                dtype=model_dtype,
            ).eval()
            pf_out = pf_wrapper(
                prefill_ids, pf_pos_ids, pf_mask, pf_ci, *pf_kcs, *pf_vcs
            )
        # pf_out = (logits [1,S,V], kc0,...,vc0,...) with kv_mode="input", return_kv=True
        pf_logits = pf_out[0]  # [1, S, V]
        pf_kcs_out = list(pf_out[1 : 1 + args.layers])
        pf_vcs_out = list(pf_out[1 + args.layers :])
        next_token = (
            pf_logits[0, -1, :].argmax(dim=-1, keepdim=True).unsqueeze(0)
        )  # [1,1]
        stage(
            "prefill_for_decode_kv",
            True,
            f"next_token={next_token.item()}  kv_filled_slots={S}",
        )

        # Step 2: build decode inputs.
        # cache_index = [S]: the decode step writes exactly one slot at position S.
        input_ids = next_token
        position_ids = torch.tensor([[S]], dtype=torch.int64)
        # Decode mask: [1, 1, 1, C]. Attends to positions 0..S (inclusive), masked beyond S.
        # attn_mask is in ex_args — traced by torch.export; must match model dtype.
        attn_mask = build_decode_mask(1, C, S, 0, dtype=model_dtype)
        cache_index = make_cache_index(S, 1, "cpu")
        # KV buffers are registered on the wrapper; the prefill KV is copied into them below.
        caches = []
        # Save a reference copy of the populated KV caches for HF reference.
        _dec_kcs_ref = [k.clone() for k in pf_kcs_out]
        _dec_vcs_ref = [v.clone() for v in pf_vcs_out]

    if args.llm_wrapper:
        if args.mode != "decode":
            print("ERROR: --llm-wrapper requires --mode decode", file=sys.stderr)
            sys.exit(1)
        wrapper = StandardLLMWrapper(
            model,
            adapter,
            args.layers,
            kv_shape,
            C,
            dtype=model_dtype,
        ).eval()
        # Inject the prefill KV state into the inner HfaExport's buffers.
        for i in range(args.layers):
            getattr(wrapper._inner, f"kc{i}").copy_(_dec_kcs_ref[i])
            getattr(wrapper._inner, f"vc{i}").copy_(_dec_vcs_ref[i])
        # The two-input example args for export.
        ex_args = (input_ids, cache_index)  # tokens=[1,1], input_pos=[1]
    else:
        wrapper = HfaExport(
            model, adapter, args.layers, kv_shape, args.kv, dtype=model_dtype
        ).eval()
        if args.mode == "decode":
            # Copy prefill KV state into the buffer-mode wrapper's registered buffers.
            for i in range(args.layers):
                getattr(wrapper, f"kc{i}").copy_(_dec_kcs_ref[i])
                getattr(wrapper, f"vc{i}").copy_(_dec_vcs_ref[i])
        ex_args = (input_ids, position_ids, attn_mask, cache_index, *caches)

    with torch.no_grad():
        eager_args = [a.clone() for a in ex_args]
        eager = wrapper(*eager_args)
        if args.mode == "prefill":
            o = ref_model(input_ids, output_hidden_states=True)
        else:  # decode: get HF reference by running two-step generation
            # Run HF prefill (with past_key_values) to get the same KV state, then
            # run HF decode (next_token) reusing that state.
            hf_pf = ref_model(prefill_ids, use_cache=True)
            o = ref_model(
                next_token.int(),
                past_key_values=hf_pf.past_key_values,
                use_cache=True,
                output_hidden_states=True,
            )
        ref = o.logits.float()

    eager_t = eager if isinstance(eager, tuple) else (eager,)
    err = (eager_t[0].float() - ref).abs().max().item()
    stage("cpu_eager_vs_HF_fp32", err < 0.1, f"max_abs_err={err:.5f}")
    # reset buffers mutated by eager run
    if args.llm_wrapper:
        # StandardLLMWrapper: restore prefill KV state in the inner HfaExport
        for i in range(args.layers):
            getattr(wrapper._inner, f"kc{i}").copy_(_dec_kcs_ref[i])
            getattr(wrapper._inner, f"vc{i}").copy_(_dec_vcs_ref[i])
    elif args.mode == "prefill":
        for i in range(args.layers):
            getattr(wrapper, f"kc{i}").zero_()
            getattr(wrapper, f"vc{i}").zero_()
    else:
        # decode: restore prefill KV state so torch.export runs with correct values
        for i in range(args.layers):
            getattr(wrapper, f"kc{i}").copy_(_dec_kcs_ref[i])
            getattr(wrapper, f"vc{i}").copy_(_dec_vcs_ref[i])

    # ---------------- export ----------------
    try:
        ep = torch.export.export(wrapper, ex_args)
        out_node = next(n for n in ep.graph.nodes if n.op == "output")
        n_outputs = (
            len(out_node.args[0]) if isinstance(out_node.args[0], (tuple, list)) else 1
        )
        stage(
            "torch.export",
            True,
            f"{sum(n.op=='call_function' for n in ep.graph.nodes)} nodes  outputs={n_outputs}",
        )
    except Exception as e:  # noqa: BLE001
        stage(
            "torch.export", False, f"{type(e).__name__}: {str(e).splitlines()[0][:300]}"
        )
        sys.exit(10)
    if args.stage == "export":
        return

    # constant_methods for the standard LLM runner: only emitted in --llm-wrapper mode.
    # get_max_seq_len  : maximum prompt length the model was exported for (S).
    # get_max_context_len : KV cache capacity (C = cache length).
    # use_kv_cache     : always True for this export path.
    # enable_dynamic_shape : always False (static shapes; no bucketing).
    # get_eos_ids      : checkpoint EOS ids (real checkpoint or --eos-ids only; the
    #                    synthetic model has no meaningful EOS).  Without it the
    #                    stock runner stops only on the tokenizer's single eos_tok().
    _constant_methods = (
        {
            "get_max_seq_len": S,
            "get_max_context_len": C,
            "use_kv_cache": True,
            "enable_dynamic_shape": False,
        }
        if args.llm_wrapper
        else None
    )
    _eos_ids = []
    if args.llm_wrapper:
        _eos_ids = (
            _detect_eos_ids(ckpt, args.eos_ids) if (args.ckpt or args.eos_ids) else []
        )
        if _eos_ids:
            _constant_methods["get_eos_ids"] = _eos_ids

    try:
        edge = to_edge(
            ep,
            constant_methods=_constant_methods,
            compile_config=EdgeCompileConfig(
                preserve_ops=[
                    torch.ops.aten.silu.default,
                    torch.ops.aten.scaled_dot_product_attention.default,
                ]
            ),
        )
        deleg, fb = _partition_report(edge.exported_program())
        stage(
            "to_edge+partition_report",
            True,
            f"delegable={len(deleg)} cpu={len(fb)} (informational) fallback_ops={sorted(set(fb))}",
        )
    except Exception as e:  # noqa: BLE001
        stage("to_edge", False, f"{type(e).__name__}: {str(e).splitlines()[0][:300]}")
        sys.exit(11)
    if args.stage == "edge":
        return

    pte_name = (
        "spyre_hfa_decode_llm.pte" if args.llm_wrapper else f"spyre_hfa_{args.mode}.pte"
    )
    try:
        partitioner = SpyrePartitioner(granularity=args.partition_granularity)
        be = edge.to_backend(partitioner)
        if partitioner.last_report is not None:
            print(
                "\n".join(format_partition_report(partitioner.last_report)), flush=True
            )
        stage(
            "SpyreBackend.preprocess(compile_fx->DeepTools)",
            True,
            f"delegates={partitioner.last_report['n_partitions'] if partitioner.last_report else '?'} "
            f"granularity={partitioner.granularity}",
        )
        et = be.to_executorch()
        if args.llm_wrapper:  # authoritative: what the serialized PTE actually exports
            assert_eos_exported(
                _eos_ids, {p.name: None for p in et.executorch_program.execution_plan}
            )
        pte = os.path.join(args.out_dir, pte_name)
        with open(pte, "wb") as f:
            f.write(et.buffer)
        stage("PTE", True, f"{len(et.buffer)} bytes")
    except Exception as e:  # noqa: BLE001
        stage(
            "to_backend/PTE",
            False,
            f"{type(e).__name__}: {str(e).splitlines()[0][:400]}",
        )
        sys.exit(12)

    # ---------------- artifacts for spyre_coverage_runner ----------------
    # For decode mode with --kv buffer: KV buffers are serialized as mutable
    # registered buffers in the PTE. ExecuTorch serializes only their shape/dtype
    # (not the initial values), so the runner always starts with zero KV caches.
    #
    # The runner-level ref.bin therefore uses the ZERO-KV-cache decode reference
    # (so the runner can validate the decode compute graph regardless of KV state).
    # The KV-populated decode correctness is validated by cpu_eager_vs_HF_fp32 above.
    man = [f"pte {pte_name}"]
    for i, a in enumerate(ex_args):
        fn = f"in{i}.bin"
        p = os.path.join(args.out_dir, fn)
        dims = " ".join(str(d) for d in a.shape)
        if a.dtype == torch.int64:
            a.contiguous().numpy().tofile(p)
            dt = "int64"
        else:
            _dump_fp16(p, a)
            dt = "fp16"
        man.append(f"input {fn} {dt} {a.dim()} {dims}")

    if args.mode == "decode":
        # Decode (KV buffers, also --llm-wrapper): the PTE's mutable KV buffers are
        # always zero-initialized at runtime (ExecuTorch does not serialize initial
        # buffer values). Generate a zero-KV-cache CPU reference so the runner can
        # validate decode graph correctness independently of prefill state.
        # The KV-populated decode correctness is validated by cpu_eager_vs_HF_fp32.
        with torch.no_grad():
            if args.llm_wrapper:
                zero_wrapper = StandardLLMWrapper(
                    model,
                    adapter,
                    args.layers,
                    kv_shape,
                    C,
                    dtype=model_dtype,
                ).eval()
                zero_out = zero_wrapper(*[a.clone() for a in ex_args])
            else:
                zero_wrapper = HfaExport(
                    model, adapter, args.layers, kv_shape, "buffer"
                ).eval()
                zero_out = zero_wrapper(*[a.clone() for a in ex_args])
        zero_logits = zero_out if isinstance(zero_out, torch.Tensor) else zero_out[0]
        # Decode output is [1, 1, V]; ref shape matches the device output shape.
        ref_out = zero_logits.half()  # [1, 1, V]
        stage(
            "zero_kv_ref_for_runner",
            True,
            f"zero-KV-cache decode logits computed for runner comparison shape={tuple(ref_out.shape)}",
        )
    else:
        # Prefill: full [1, S, V] logit tensor so the runner compares the whole
        # sequence output (matches what the device delegate returns).
        ref_out = ref.half()  # [1, S, V]

    _dump_fp16(os.path.join(args.out_dir, "ref.bin"), ref_out)
    dims_str = " ".join(str(d) for d in ref_out.shape)
    man.append(f"output ref.bin fp16 {ref_out.dim()} {dims_str}")
    open(os.path.join(args.out_dir, "manifest.txt"), "w").write("\n".join(man) + "\n")
    print(
        f"artifacts: {args.out_dir}  run: cd {args.out_dir} && $EXECUTORCH_SPYRE_COVERAGE_RUNNER ."
    )


if __name__ == "__main__":
    main()
