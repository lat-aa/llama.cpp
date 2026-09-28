#!/usr/bin/env python3
"""
RAG-stack measurable gates for llama-server (embed + looped gen + e2e).

Subcommands:
  embed       - Gate A/B: embed QPS + optional cosine vs --ref
  loop-depth  - Gate C: require GGUF num_loops / loop_count
  gen-perf    - Gate D: llama-bench arms (baseline / FA / FA+q8 KV)
  e2e         - Gate F: toy corpus hit@k + optional chat latency
  sparse      - Gate E: FlagEmbedding sparse Spearman (optional)

Examples:
  python rag_stack_bench.py embed --url http://127.0.0.1:8080 --out embed.json
  python rag_stack_bench.py loop-depth -m /path/model.gguf --require
  python rag_stack_bench.py gen-perf --bin ./build/bin/Release/llama-bench.exe -m /path/nb.gguf
  python rag_stack_bench.py e2e --embed-url http://127.0.0.1:8080 --gen-url http://127.0.0.1:8081
  python rag_stack_bench.py sparse --hf BAAI/bge-m3
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------


def write_json(path: str, obj: Any) -> None:
    text = json.dumps(obj, indent=2, ensure_ascii=False)
    print(text)
    if path:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text + "\n")


def load_json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / (na * nb)


def l2_normalize(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def _requests():
    try:
        import requests
    except ImportError:
        print("pip install requests", file=sys.stderr)
        raise SystemExit(1)
    return requests


def embed_batch(url: str, texts: list[str], *, normalize: bool = False) -> list[list[float]]:
    requests = _requests()
    r = requests.post(
        f"{url.rstrip('/')}/v1/embeddings",
        json={"input": texts, "encoding_format": "float"},
        timeout=600,
    )
    r.raise_for_status()
    data = sorted(r.json()["data"], key=lambda x: x["index"])
    vecs = [row["embedding"] for row in data]
    if normalize:
        return [l2_normalize(v) for v in vecs]
    return vecs


def extract_json_payload(stdout: str) -> Any:
    """Parse trailing JSON object or array from tool logs mixed with text."""
    s = stdout.rstrip()
    if not s:
        return None
    # Prefer array (llama-bench -o json often emits [{pp},{tg},...])
    for opener, closer in (("[", "]"), ("{", "}")):
        start = s.rfind(opener)
        if start < 0:
            continue
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(s[start : i + 1])
                    except json.JSONDecodeError:
                        break
    return None


def spearman(xs: list[float], ys: list[float]) -> float:
    def rank(vals: list[float]) -> list[float]:
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        r = [0.0] * len(vals)
        for rank_i, i in enumerate(order):
            r[i] = float(rank_i)
        return r

    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    denx = sum((a - mx) ** 2 for a in rx) ** 0.5
    deny = sum((b - my) ** 2 for b in ry) ** 0.5
    if denx <= 0 or deny <= 0:
        return 0.0
    return num / (denx * deny)


# ---------------------------------------------------------------------------
# GGUF loop-depth (gate C)
# ---------------------------------------------------------------------------

LOOP_KEYS = ("nanbeige.num_loops", "nanbeige.loop_count", "general.num_loops")


def _gguf_kv_via_gguf(path: str, key: str) -> int | None:
    try:
        from gguf import GGUFReader
    except Exception:
        return None
    reader = GGUFReader(path)
    for field in reader.fields.values():
        if field.name != key:
            continue
        try:
            val = field.contents()
            if isinstance(val, (list, tuple)):
                val = val[0]
            return int(val)
        except Exception:
            return None
    return None


def _gguf_kv_via_scan(path: str, key: str) -> int | None:
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            raise SystemExit(f"not a GGUF: {path}")
        version = struct.unpack("<I", f.read(4))[0]
        if version < 2:
            raise SystemExit(f"unsupported GGUF version {version}")
        _n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]

        def skip_value(typ: int) -> None:
            if typ in (0, 1, 8):
                f.read(1)
            elif typ in (2, 3, 5):
                f.read(2)
            elif typ in (4, 6, 7):
                f.read(4)
            elif typ in (11, 12, 13):
                f.read(8)
            elif typ == 9:
                sl = struct.unpack("<Q", f.read(8))[0]
                if sl > 1 << 30:
                    raise SystemExit(f"implausible string length {sl}")
                f.read(sl)
            elif typ == 10:
                at = struct.unpack("<I", f.read(4))[0]
                al = struct.unpack("<Q", f.read(8))[0]
                if al > 1 << 28:
                    raise SystemExit(f"implausible array length {al}")
                for _ in range(al):
                    skip_value(at)
            else:
                raise SystemExit(f"unsupported kv type {typ}")

        for _ in range(n_kv):
            klen = struct.unpack("<Q", f.read(8))[0]
            if klen > 1 << 20:
                raise SystemExit(f"implausible key length {klen}")
            k = f.read(klen).decode("utf-8", errors="replace")
            typ = struct.unpack("<I", f.read(4))[0]
            if typ == 4:
                val = struct.unpack("<I", f.read(4))[0]
                if k == key:
                    return val
            elif typ == 6:
                val = struct.unpack("<i", f.read(4))[0]
                if k == key:
                    return int(val)
            else:
                skip_value(typ)
    return None


def read_gguf_kv_u32(path: str, key: str) -> int | None:
    try:
        from gguf import GGUFReader  # noqa: F401

        # Prefer gguf-py; do not fall through to scanner on a miss.
        return _gguf_kv_via_gguf(path, key)
    except Exception:
        pass
    return _gguf_kv_via_scan(path, key)


# ---------------------------------------------------------------------------
# subcommands
# ---------------------------------------------------------------------------

EMBED_PROMPTS = [
    "What is BGE-M3?",
    "Definition of BM25",
    "短文本嵌入吞吐测试",
    "多语言检索样例句子",
    "machine learning retrieval",
    "向量检索与重排序",
    "The quick brown fox",
    "llama.cpp embeddings server",
] * 4


def cmd_embed(args: argparse.Namespace) -> int:
    for _ in range(args.warmup):
        embed_batch(args.url, EMBED_PROMPTS[: args.batch_size])

    t0 = time.perf_counter()
    n_texts = 0
    last_vecs: list[list[float]] = []
    for _ in range(args.rounds):
        for i in range(0, len(EMBED_PROMPTS), args.batch_size):
            chunk = EMBED_PROMPTS[i : i + args.batch_size]
            last_vecs = embed_batch(args.url, chunk)
            n_texts += len(chunk)
    elapsed = time.perf_counter() - t0
    qps = n_texts / elapsed if elapsed > 0 else 0.0

    summary: dict[str, Any] = {
        "gate": "A/B",
        "url": args.url,
        "n_texts": n_texts,
        "elapsed_s": elapsed,
        "embed_qps": qps,
        "batch_size": args.batch_size,
        "rss_mb_note": "sample server process RSS externally for gate A",
        "n_dim": len(last_vecs[0]) if last_vecs else 0,
        "sample_vecs": last_vecs[:2],
    }

    gate_ok = True
    if args.ref:
        ref = load_json(args.ref)
        ref_vecs = ref.get("sample_vecs") or []
        cosines = []
        for a, b in zip(summary["sample_vecs"], ref_vecs):
            if a and b and len(a) == len(b):
                cosines.append(cosine(a, b))
        summary["cos_vs_ref"] = cosines
        summary["cos_vs_ref_min"] = min(cosines) if cosines else None
        summary["gate_quality_pass"] = bool(cosines) and min(cosines) >= 0.999
        gate_ok = bool(summary["gate_quality_pass"])

    write_json(args.out, summary)
    if args.ref and not gate_ok:
        print("FAIL: cosine vs --ref < 0.999", file=sys.stderr)
        return 1
    return 0


def cmd_loop_depth(args: argparse.Namespace) -> int:
    path = str(Path(args.model))
    for key in LOOP_KEYS:
        val = read_gguf_kv_u32(path, key)
        if val is not None:
            print(f"{key}={val}")
            if args.require and val != args.expect:
                print(f"FAIL: expected {args.expect}", file=sys.stderr)
                return 1
            return 0

    print("num_loops key MISSING")
    if args.require:
        print("FAIL: nanbeige.num_loops (or loop_count) required (gate C)", file=sys.stderr)
        return 1
    return 2


def _bench_metrics(payload: Any) -> dict[str, float | None]:
    """Extract pp/tg avg_ts from llama-bench JSON object or list of objects."""
    rows: list[dict[str, Any]]
    if isinstance(payload, list):
        rows = [x for x in payload if isinstance(x, dict)]
    elif isinstance(payload, dict):
        rows = [payload]
    else:
        rows = []

    pp = tg = None
    for row in rows:
        n_prompt = int(row.get("n_prompt") or 0)
        n_gen = int(row.get("n_gen") or 0)
        avg_ts = row.get("avg_ts")
        if avg_ts is None:
            continue
        avg_ts = float(avg_ts)
        if n_prompt > 0 and n_gen == 0:
            pp = avg_ts
        elif n_gen > 0 and n_prompt == 0:
            tg = avg_ts
        elif n_gen > 0:
            # combined row: prefer as tg if only one sample shape
            tg = avg_ts if tg is None else tg
            if n_prompt > 0 and pp is None:
                pp = avg_ts
    return {"pp_avg_ts": pp, "tg_avg_ts": tg}


def cmd_gen_perf(args: argparse.Namespace) -> int:
    arms = {
        "baseline_no_fa_f16kv": ["-fa", "0"],
        "fa": ["-fa", "1"],
        "fa_q8kv": ["-fa", "1", "-ctk", "q8_0", "-ctv", "q8_0"],
    }
    summary: dict[str, Any] = {"gate": "D", "model": args.model, "arms": {}}
    any_fail = False

    for name, extra in arms.items():
        cmd = [
            args.bin,
            "-m",
            args.model,
            "-p",
            "512",
            "-n",
            "128",
            "-r",
            "3",
            "-o",
            "json",
            *extra,
        ]
        print("+", " ".join(cmd), flush=True)
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            any_fail = True
            summary["arms"][name] = {
                "ok": False,
                "cmd": cmd,
                "stderr": (p.stderr or "")[-2000:],
                "stdout": (p.stdout or "")[-2000:],
            }
            continue
        payload = extract_json_payload(p.stdout or "")
        metrics = _bench_metrics(payload)
        summary["arms"][name] = {
            "ok": True,
            "cmd": cmd,
            "metrics": metrics,
            "result": payload,
        }

    base = summary["arms"].get("baseline_no_fa_f16kv", {})
    fa = summary["arms"].get("fa", {})
    q8 = summary["arms"].get("fa_q8kv", {})

    def tg(arm: dict) -> float | None:
        m = arm.get("metrics") or {}
        return m.get("tg_avg_ts")

    def pp(arm: dict) -> float | None:
        m = arm.get("metrics") or {}
        return m.get("pp_avg_ts")

    b_tg, f_tg, q_tg = tg(base), tg(fa), tg(q8)
    b_pp, f_pp = pp(base), pp(fa)

    fa_tg_lift = (f_tg / b_tg - 1.0) if (f_tg and b_tg) else None
    fa_pp_lift = (f_pp / b_pp - 1.0) if (f_pp and b_pp) else None
    q8_tg_reg = (q_tg / f_tg - 1.0) if (q_tg and f_tg) else None

    gate_fa = False
    if fa_tg_lift is not None and fa_tg_lift >= 0.10:
        gate_fa = True
    if fa_pp_lift is not None and fa_pp_lift >= 0.15:
        gate_fa = True

    gate_q8 = q8_tg_reg is not None and q8_tg_reg >= -0.05

    summary["gates"] = {
        "fa_tg_lift": fa_tg_lift,
        "fa_pp_lift": fa_pp_lift,
        "q8kv_tg_regress": q8_tg_reg,
        "gate_fa_pass": gate_fa,
        "gate_q8kv_pass": gate_q8,
        "note": "VRAM -20% only when GPU JSON exposes it; CPU builds skip VRAM",
    }

    write_json(args.out, summary)
    if any_fail:
        print("FAIL: one or more llama-bench arms failed", file=sys.stderr)
        return 1
    if args.require_gates and not gate_fa:
        print("FAIL: FA gate (need +10% tg or +15% pp)", file=sys.stderr)
        return 1
    return 0


CORPUS = [
    ("d0", "BGE-M3 supports dense sparse and multi-vector retrieval."),
    ("d1", "Nanbeige4.2-3B uses looped transformer layers with num_loops=2."),
    ("d2", "llama.cpp serves embeddings via /v1/embeddings."),
    ("d3", "Flash attention reduces attention latency on GPU."),
    ("d4", "Quantized KV cache lowers VRAM for long context."),
    ("d5", "Radix prefix cache shares KV across concurrent sessions."),
    ("d6", "BM25 is a classic lexical ranking function."),
    ("d7", "ColBERT uses late interaction over token vectors."),
    ("d8", "Rerankers reorder dense-retrieved candidates."),
    ("d9", "Greedy decoding with temperature 0 is deterministic."),
]

QUERIES = [
    ("What does BGE-M3 support?", ["d0"]),
    ("How does Nanbeige4.2 get depth 44?", ["d1"]),
    ("Where are embeddings served?", ["d2"]),
    ("How to save VRAM on long context?", ["d4"]),
    ("What is ColBERT?", ["d7"]),
]


def cmd_e2e(args: argparse.Namespace) -> int:
    requests = _requests()
    doc_ids = [did for did, _ in CORPUS]
    doc_text = {did: text for did, text in CORPUS}

    t_emb0 = time.perf_counter()
    doc_vecs = embed_batch(args.embed_url, [t for _, t in CORPUS], normalize=True)
    q_vecs = embed_batch(args.embed_url, [q for q, _ in QUERIES], normalize=True)
    emb_ms = (time.perf_counter() - t_emb0) * 1000

    hits = 0
    rows: list[dict[str, Any]] = []
    top_by_query: list[list[str]] = []
    for (query, relevant), qv in zip(QUERIES, q_vecs):
        scored = [(cosine(qv, dv), did) for did, dv in zip(doc_ids, doc_vecs)]
        scored.sort(reverse=True)
        top = [did for _, did in scored[: args.k]]
        top_by_query.append(top)
        hit = any(r in top for r in relevant)
        hits += int(hit)
        rows.append({"query": query, "top": top, "hit": hit})

    hit_at_k = hits / len(QUERIES)
    e2e_ms: list[float] = []
    sample_ans = None

    if args.gen_url:
        for i, ((query, _), top) in enumerate(zip(QUERIES, top_by_query)):
            ctx = "\n".join(f"- {doc_text[did]}" for did in top)
            prompt = f"Context:\n{ctx}\n\nQuestion: {query}\nAnswer briefly."
            t1 = time.perf_counter()
            r = requests.post(
                f"{args.gen_url.rstrip('/')}/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 32,
                    "temperature": 0.0,
                },
                timeout=300,
            )
            r.raise_for_status()
            ans = r.json()["choices"][0]["message"].get("content") or ""
            dt = (time.perf_counter() - t1) * 1000
            e2e_ms.append(dt + emb_ms / max(len(QUERIES), 1))
            if i == 0:
                sample_ans = ans

    e2e_sorted = sorted(e2e_ms)
    e2e_p50 = e2e_sorted[len(e2e_sorted) // 2] if e2e_sorted else None

    summary: dict[str, Any] = {
        "gate": "F",
        "hit_at_k": hit_at_k,
        "k": args.k,
        "n_queries": len(QUERIES),
        "embed_total_ms": emb_ms,
        "e2e_p50_ms": e2e_p50,
        "e2e_samples_ms": e2e_ms,
        "sample_ans": sample_ans,
        "rows": rows,
        "gate_f_note": "vs baseline: hit@k drop <= 1pp; with rerank expect +5pp; e2e p50 <= 0.85x",
    }
    write_json(args.out, summary)

    if args.require and hit_at_k < args.min_hit:
        print(f"FAIL: hit_at_k={hit_at_k:.3f} < min_hit={args.min_hit}", file=sys.stderr)
        return 1
    return 0


def cmd_sparse(args: argparse.Namespace) -> int:
    summary: dict[str, Any] = {
        "gate": "E",
        "hf": args.hf,
        "gate_e_accept": "Spearman>=0.95 on overlapped top-32 sparse weights vs FlagEmbedding",
        "status": "skipped",
    }

    try:
        from FlagEmbedding import BGEM3FlagModel  # type: ignore
    except Exception as e:
        summary["status"] = "flagembedding_missing"
        summary["error"] = str(e)
        summary["note"] = "Install FlagEmbedding to run gate E; do not merge sparse without this number."
        write_json(args.out, summary)
        return 0

    model = BGEM3FlagModel(args.hf, use_fp16=True)
    hf_out = model.encode(args.queries, return_dense=False, return_sparse=True, return_colbert_vecs=False)

    summary["hf_sparse_sample"] = []
    for q, w in zip(args.queries, hf_out["lexical_weights"]):
        items = sorted(w.items(), key=lambda kv: -float(kv[1]))[:32]
        summary["hf_sparse_sample"].append({"query": q, "top": items})

    if not args.gguf_url:
        summary["status"] = "hf_only_baseline"
        summary["note"] = "Start server with sparse path and re-run with --gguf-url to score Spearman."
        write_json(args.out, summary)
        return 0

    try:
        requests = _requests()
        r = requests.post(
            f"{args.gguf_url.rstrip('/')}/v1/embeddings",
            json={"input": args.queries, "sparse": True},
            timeout=300,
        )
        if r.status_code != 200:
            summary["status"] = "server_no_sparse"
            summary["http"] = r.status_code
            summary["body"] = r.text[:500]
            write_json(args.out, summary)
            return 0

        body = r.json()
        spears: list[float] = []
        for i, _q in enumerate(args.queries):
            hf = dict(summary["hf_sparse_sample"][i]["top"])
            spar = body["data"][i].get("sparse_weights") or body["data"][i].get("sparse") or {}
            if isinstance(spar, list):
                spar = {str(x.get("token", x.get("id"))): float(x["weight"]) for x in spar}
            keys = [k for k in hf.keys() if k in spar]
            if len(keys) < 5:
                continue
            spears.append(spearman([float(hf[k]) for k in keys], [float(spar[k]) for k in keys]))
        summary["spearman"] = spears
        summary["spearman_min"] = min(spears) if spears else None
        summary["gate_e_pass"] = bool(spears) and min(spears) >= args.min_spearman
        summary["status"] = "compared"
    except Exception as e:
        summary["status"] = "compare_error"
        summary["error"] = str(e)
        write_json(args.out, summary)
        return 0

    write_json(args.out, summary)
    if summary["status"] == "compared" and not summary.get("gate_e_pass"):
        print(
            f"FAIL: spearman_min={summary.get('spearman_min')} < {args.min_spearman}",
            file=sys.stderr,
        )
        return 1
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)

    p_embed = sp.add_parser("embed", help="Gate A/B: embed QPS + cosine vs --ref")
    p_embed.add_argument("--url", default="http://127.0.0.1:8080")
    p_embed.add_argument("--warmup", type=int, default=2)
    p_embed.add_argument("--rounds", type=int, default=10)
    p_embed.add_argument("--batch-size", type=int, default=8)
    p_embed.add_argument("--out", default="")
    p_embed.add_argument("--ref", default="", help="previous JSON for cosine gate")
    p_embed.set_defaults(func=cmd_embed)

    p_loop = sp.add_parser("loop-depth", help="Gate C: GGUF num_loops / loop_count")
    p_loop.add_argument("-m", "--model", required=True)
    p_loop.add_argument("--expect", type=int, default=2)
    p_loop.add_argument("--require", action="store_true")
    p_loop.set_defaults(func=cmd_loop_depth)

    p_gen = sp.add_parser("gen-perf", help="Gate D: llama-bench FA / q8 KV arms")
    p_gen.add_argument("--bin", required=True, help="path to llama-bench")
    p_gen.add_argument("-m", "--model", required=True)
    p_gen.add_argument("--out", default="gen_perf.json")
    p_gen.add_argument("--require-gates", action="store_true", help="exit 1 if FA gate fails")
    p_gen.set_defaults(func=cmd_gen_perf)

    p_e2e = sp.add_parser("e2e", help="Gate F: hit@k + optional chat e2e")
    p_e2e.add_argument("--embed-url", required=True)
    p_e2e.add_argument("--gen-url", default="")
    p_e2e.add_argument("--k", type=int, default=3)
    p_e2e.add_argument("--out", default="rag_e2e.json")
    p_e2e.add_argument("--require", action="store_true")
    p_e2e.add_argument("--min-hit", type=float, default=1.0)
    p_e2e.set_defaults(func=cmd_e2e)

    p_sp = sp.add_parser("sparse", help="Gate E: FlagEmbedding sparse Spearman")
    p_sp.add_argument("--gguf-url", default="")
    p_sp.add_argument("--hf", default="BAAI/bge-m3")
    p_sp.add_argument("--out", default="sparse_parity.json")
    p_sp.add_argument("--queries", nargs="*", default=["What is BGE-M3?", "BM25 definition"])
    p_sp.add_argument("--min-spearman", type=float, default=0.95)
    p_sp.set_defaults(func=cmd_sparse)

    args = ap.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
