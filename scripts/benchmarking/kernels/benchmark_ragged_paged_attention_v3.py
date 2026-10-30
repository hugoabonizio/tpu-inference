# Copyright 2026 Google LLC
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
"""Benchmark RPA v3 on one chip: one TensorCore vs all of them.

On megacore chips (v4, v5p) the kernel splits its grid across the chip's
TensorCores. This times one attention layer with num_cores=1 (one core does
all the work, the schedule before the split) against the default for each
--core-split, and checks that every run returns the same output and KV cache
bit for bit. Inputs are built like the runner's (decode-only sequences first,
everything else in the mixed range); the cases are

  decode_b1@100k    one decode sequence at 100k of context
  decode_b{B}       B decode sequences, kv lens uniform in [20k, 120k)
  prefill_8k@{N}k   one 8k-token chunk ending at N k of context
  mixed             15 decodes + an 8k chunk at 64k

Run on a TPU host (one chip is enough):
    python scripts/benchmarking/kernels/benchmark_ragged_paged_attention_v3.py \\
        --model qwen3.5-4b,llama3.1-8b --kv-dtype bfloat16,float8_e4m3fn

The output is a markdown table, one row per (model, KV dtype, case).
"""

import argparse
import functools
import importlib.metadata
import json
import time

import jax
import jax.numpy as jnp
import numpy as np

from tpu_inference.kernels.ragged_paged_attention.v3 import kernel as rpa
from tpu_inference.kernels.ragged_paged_attention.v3.util import cdiv

# (num_q_heads, num_kv_heads, head_dim, page_size) of the attention layers.
MODELS = {
    "qwen3.5-4b": (16, 4, 256, 256),
    "llama3.1-8b": (32, 8, 128, 128),
}
MAX_MODEL_LEN = 131072
MAX_NUM_SEQS = 64


def make_cases(decode_batches):
    """[(name, [(q_len, kv_len)], max_num_tokens)] in run order."""
    rng = np.random.default_rng(1)
    cases = [("decode_b1@100k", [(1, 100000)], 128)]
    for b in decode_batches:
        lens = rng.integers(20000, 120000, b)
        cases.append((f"decode_b{b}", [(1, int(n)) for n in lens], 128))
    for end in (8192, 32768, 65536, 106496):
        cases.append((f"prefill_8k@{end // 1024}k", [(8192, end)], 8192))
    lens = rng.integers(20000, 120000, 15)
    cases.append(
        ("mixed", [(1, int(n)) for n in lens] + [(8192, 65536)], 8320))
    return cases


def causal_tflop(shape, seqs):
    """Attention FLOPs (QK^T and PV) of the causal prefill/mixed rows."""
    num_q_heads, _, head_dim, _ = shape
    flops = 0
    for q_len, kv_len in seqs:
        if q_len > 1:
            avg_ctx = kv_len - q_len + (q_len + 1) / 2
            flops += 4 * q_len * avg_ctx * num_q_heads * head_dim
    return flops / 1e12


def make_inputs(shape, seqs, kv_dtype, max_num_tokens):
    """Kernel inputs except the cache; the cache comes from make_cache."""
    num_q_heads, num_kv_heads, head_dim, page_size = shape
    rng = np.random.default_rng(0)
    need = [cdiv(kv_len, page_size) for _, kv_len in seqs]
    num_pages = sum(need) + 16
    pages_per_seq = MAX_MODEL_LEN // page_size
    perm = rng.permutation(num_pages).astype(np.int32)
    page_indices = np.zeros((MAX_NUM_SEQS, pages_per_seq), np.int32)
    used = 0
    for i, n in enumerate(need):
        page_indices[i, :n] = perm[used:used + n]
        used += n
    kv_lens = np.zeros(MAX_NUM_SEQS, np.int32)
    kv_lens[:len(seqs)] = [kv_len for _, kv_len in seqs]
    cu_q_lens = np.zeros(MAX_NUM_SEQS + 1, np.int32)
    cu_q_lens[1:len(seqs) + 1] = np.cumsum([q_len for q_len, _ in seqs])
    cu_q_lens[len(seqs) + 1:] = cu_q_lens[len(seqs)]
    num_decode = next((i for i, (q_len, _) in enumerate(seqs) if q_len != 1),
                      len(seqs))
    distribution = np.array([num_decode, num_decode, len(seqs)], np.int32)
    kq, kk, kv = jax.random.split(jax.random.key(1), 3)
    q = jax.random.normal(kq, (max_num_tokens, num_q_heads, head_dim),
                          jnp.bfloat16)
    k = (jax.random.normal(kk, (max_num_tokens, num_kv_heads, head_dim),
                           jnp.bfloat16) * 0.5).astype(kv_dtype)
    v = (jax.random.normal(kv, (max_num_tokens, num_kv_heads, head_dim),
                           jnp.bfloat16) * 0.5).astype(kv_dtype)
    cache_shape = rpa.get_kv_cache_shape(num_pages, page_size, num_kv_heads,
                                         head_dim, kv_dtype)
    return (q, k, v, jnp.asarray(kv_lens),
            jnp.asarray(page_indices.reshape(-1)), jnp.asarray(cu_q_lens),
            jnp.asarray(distribution)), cache_shape


@functools.partial(jax.jit, static_argnums=(0, 1))
def make_cache(cache_shape, kv_dtype):
    """The same cache contents on every call."""
    cache = jax.random.normal(jax.random.key(2), cache_shape, jnp.bfloat16)
    return (cache * 0.5).astype(kv_dtype)


def build_step(sm_scale, num_cores, core_split, d_block_sizes, calls):
    """`calls` chained kernel calls, the cache donated and threaded through
    (as in serving), so no copy of the cache is timed."""

    def step(q, k, v, cache, kv_lens, page_indices, cu_q_lens, distribution):
        for _ in range(calls):
            out, cache = rpa.ragged_paged_attention(
                q,
                k,
                v,
                cache,
                kv_lens,
                page_indices,
                cu_q_lens,
                distribution,
                sm_scale=sm_scale,
                num_cores=num_cores,
                core_split=core_split,
                d_block_sizes=d_block_sizes)
        return out, cache

    return jax.jit(step, donate_argnums=3)


def time_kernel(step, inputs, cache, calls, min_seconds):
    """Mean time (us) per kernel call over >= min_seconds of back-to-back
    steps, after a compile/warm-up step; also the last output and cache."""
    q, k, v, *meta = inputs
    out, cache = jax.block_until_ready(step(q, k, v, cache, *meta))
    t0 = time.perf_counter()
    out, cache = jax.block_until_ready(step(q, k, v, cache, *meta))
    once = time.perf_counter() - t0
    steps = max(3, int(min_seconds / max(once, 1e-6)))
    t0 = time.perf_counter()
    for _ in range(steps):
        out, cache = step(q, k, v, cache, *meta)
    jax.block_until_ready((out, cache))
    elapsed = time.perf_counter() - t0
    return elapsed / (steps * calls) * 1e6, out, cache


def bench_case(model, kv_dtype, case, splits, args):
    name, seqs, max_num_tokens = case
    shape = MODELS[model]
    inputs, cache_shape = make_inputs(shape, seqs, kv_dtype, max_num_tokens)
    sm_scale = shape[2]**-0.5
    num_tokens = sum(q_len for q_len, _ in seqs)
    row = {
        "model": model,
        "kv_dtype": jnp.dtype(kv_dtype).name,
        "case": name,
        "d_block_sizes": None,
    }
    arms = [("1 core", 1, "seq")] + [(s, None, s) for s in splits]
    for attempt in range(2):
        try:
            ref_out = ref_cache = None
            times, identical = {}, True
            for arm, num_cores, split in arms:
                step = build_step(sm_scale, num_cores, split,
                                  row["d_block_sizes"], args.calls)
                t, out, cache = time_kernel(step, inputs,
                                            make_cache(cache_shape, kv_dtype),
                                            args.calls, args.min_seconds)
                times[arm] = t
                if ref_out is None:
                    ref_out, ref_cache = out, cache
                else:
                    identical &= bool(
                        jnp.array_equal(out[:num_tokens], ref_out[:num_tokens])
                        and jnp.array_equal(cache, ref_cache))
                del out, cache
            break
        except Exception as e:  # noqa: BLE001
            # The decode launch is compiled for every batch, and its default
            # blocks can overflow VMEM (fp8 KV with head_dim 256 on v5p);
            # retry every arm with smaller decode blocks.
            if attempt or args.fallback_d_block_sizes is None:
                row["error"] = " ".join(str(e).split())[:200]
                return row
            row["d_block_sizes"] = args.fallback_d_block_sizes
    row["us"] = times
    row["identical"] = identical
    tflop = causal_tflop(shape, seqs)
    if tflop:
        row["tflops"] = {arm: tflop / (t * 1e-6) for arm, t in times.items()}
    return row


def format_row(row, splits):
    cells = [row["model"], row["kv_dtype"], row["case"]]
    if "error" in row:
        return "| " + " | ".join(cells) + f" | FAIL: {row['error']} |"
    if row["d_block_sizes"]:
        cells[2] += " †"
    base = row["us"]["1 core"]
    cells.append(f"{base:.0f}")
    for s in splits:
        cells += [f"{row['us'][s]:.0f}", f"{base / row['us'][s]:.2f}x"]
    if "tflops" in row:
        cells.append(" → ".join(f"{row['tflops'][a]:.0f}"
                                for a in ("1 core", splits[0])))
    else:
        cells.append("")
    cells.append("yes" if row["identical"] else "NO")
    return "| " + " | ".join(cells) + " |"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",
                        default="qwen3.5-4b",
                        help="comma-separated, from: " + ", ".join(MODELS))
    parser.add_argument("--kv-dtype", default="bfloat16,float8_e4m3fn")
    parser.add_argument(
        "--core-split",
        default="auto",
        help="comma-separated core splits to time against one core "
        "(auto, seq, kv_head)")
    parser.add_argument("--cases",
                        default=None,
                        help="comma-separated case-name prefixes to run "
                        "(default: all)")
    parser.add_argument("--decode-batches", default="2,4,8,32")
    parser.add_argument("--calls",
                        type=int,
                        default=4,
                        help="kernel calls chained in one jitted step")
    parser.add_argument("--min-seconds",
                        type=float,
                        default=1.0,
                        help="timed wall time per measurement")
    parser.add_argument(
        "--fallback-d-block-sizes",
        default="1,2048,1,2048",
        help="decode block sizes to retry with when the default ones do not "
        "compile (rows marked †); 'none' to report the failure instead")
    parser.add_argument("--json", default=None, help="also write rows here")
    args = parser.parse_args()
    args.fallback_d_block_sizes = (
        None if args.fallback_d_block_sizes == "none" else tuple(
            int(x) for x in args.fallback_d_block_sizes.split(",")))
    splits = args.core_split.split(",")
    cases = make_cases([int(b) for b in args.decode_batches.split(",")])
    if args.cases:
        prefixes = tuple(args.cases.split(","))
        cases = [c for c in cases if c[0].startswith(prefixes)]

    dev = jax.devices()[0]
    assert dev.platform == "tpu", "requires a TPU host"
    try:
        libtpu = importlib.metadata.version("libtpu")
    except importlib.metadata.PackageNotFoundError:
        libtpu = "?"
    print(f"{dev.device_kind}, num_cores={getattr(dev, 'num_cores', 1)}, "
          f"jax {jax.__version__}, libtpu {libtpu}; time per kernel call "
          f"in us, speedup vs 1 core\n")
    header = ["model", "KV", "case", "1 core"]
    for s in splits:
        header += [s, f"{s} x"]
    header += ["TFLOP/s", "bit-identical"]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header), flush=True)
    rows = []
    for model in args.model.split(","):
        for kv_dtype in args.kv_dtype.split(","):
            for case in cases:
                row = bench_case(model, jnp.dtype(kv_dtype), case, splits,
                                 args)
                rows.append(row)
                print(format_row(row, splits), flush=True)
    if any(row.get("d_block_sizes") for row in rows):
        print(f"\n† the default decode blocks do not compile (the decode "
              f"launch is built for every batch); every column uses "
              f"d_block_sizes={args.fallback_d_block_sizes}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
