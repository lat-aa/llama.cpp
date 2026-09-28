# RAG-stack measurable baselines

Evidence package for gates A-F. Fill numbers on the target machine; do not claim fixes without before/after JSON.

Harness: `python tools/server/bench/rag-stack/rag_stack_bench.py <subcommand> ...`

## Machine lock (this host)

| Field | Value |
|------|-------|
| CPU | 11th Gen Intel Core i7-11700 @ 2.50GHz |
| Backend | CPU only (no GPU build) |
| Nanbeige GGUF | `E:\data\USearch\.config\models\Nanbeige4.2-3B-Q4_K_M.gguf.broken` |
| GGUF layout | `nanbeige.loop_count=2`, `block_count=44` (logical), tensors `blk.0..21` only |
| Build | `b11228-1095fef3b` Release |
| BGE GGUF | not available locally (gates A/B/E/F blocked) |

## Locked commands

### Dense embed (gates A/B)

```bash
llama-server -m /path/bge-m3.gguf --embedding --pooling cls -fa on \
  -ub 2048 -b 2048 -np 8 --embd-normalize 2 --port 8080

python tools/server/bench/rag-stack/rag_stack_bench.py embed \
  --url http://127.0.0.1:8080 --out embed_before.json
```

RSS for gate A: sample the server process externally (harness does not remote-RSS).

### Looped gen (gates C/D)

```bash
python tools/server/bench/rag-stack/rag_stack_bench.py loop-depth -m /path/nanbeige.gguf --require

python tools/server/bench/rag-stack/rag_stack_bench.py gen-perf \
  --bin ./build/bin/Release/llama-bench.exe \
  --model /path/nanbeige42-3b-Q4_K_M.gguf --out gen_before.json
```

### RAG e2e (gate F, nightly)

```bash
python tools/server/bench/rag-stack/rag_stack_bench.py e2e \
  --embed-url http://127.0.0.1:8080 --gen-url http://127.0.0.1:8081 \
  --out rag_before.json
```

### Sparse parity (gate E, optional)

```bash
python tools/server/bench/rag-stack/rag_stack_bench.py sparse \
  --gguf-url http://127.0.0.1:8080 --hf BAAI/bge-m3 --out sparse_before.json
```

## Expected pre-fix symptoms

| Area | Symptom |
|------|---------|
| Embed RSS | Host RSS inflated by ~`n_vocab * n_ubatch * 4` logits buffer (#29388) |
| Embed CLS | All tokens marked output; last-layer FFN not CLS-only |
| Looped gen | Missing `num_loops` GGUF key silently runs depth/2 |
| Looped KV | Logical layers = phys * loops => ~2x KV vs naive 22-layer |

## Gate accept lines

| Gate | Pass |
|------|------|
| A | RSS drop >= 30% or >= 0.7 * vocab * ubatch * 4 bytes; cos >= 0.999 |
| B | QPS +15% or measurable last-layer cut; cos >= 0.999; MEAN unchanged |
| C | Missing num_loops => load fail; good GGUF loads |
| D | FA tg/pp lift; FA+q8 KV VRAM -20% with tg regress <= 5%; ngram optional +20% tok/s |
| E | Spearman >= 0.95 vs FlagEmbedding sparse |
| F | hit@10 not worse than -1pp; rerank +5pp to recommend; e2e p50 <= 0.85x |

## Evidence table

```text
| gate | before | after | pass? | cmd / commit |
| A RSS_MB | n/a (no BGE GGUF) | | blocked | |
| B qps | n/a | | blocked | |
| C load_missing_loops | fail: missing num_loops | loads via loop_count=2, phys=22, logical=44 | pass | loop-depth + llama-cli smoke |
| D tg (CPU) | fa0: 6.79 t/s | fa1: 7.82 (+15.1%); fa+q8kv: 7.63 (-2.4% vs fa) | pass FA tg; VRAM n/a on CPU | gen-perf |
| E spearman | | | blocked | need BGE + FlagEmbedding |
| F hit@10 / p50_ms | | | blocked | need BGE + server pair |
```

### Gate C detail (this GGUF)

- Metadata: `nanbeige.loop_count=2`, `block_count=44`, tensors only `blk.0`..`blk.21`.
- Loader accepts `loop_count` as alias; treats `block_count` as logical when alias is used.
- Smoke: `llama-cli -m ...broken -p hi -n 1 -c 256` loads and generates.

### Gate D detail (CPU, tg128, r=3)

| Arm | avg_ts (tok/s) | vs baseline | vs FA |
|-----|----------------|-------------|-------|
| baseline `-fa 0` f16kv | 6.790 | - | - |
| `-fa 1` | 7.816 | +15.1% | - |
| `-fa 1 -ctk/ctv q8_0` | 7.626 | +12.3% | -2.4% |
| VRAM | n/a | CPU build, no GPU | |
