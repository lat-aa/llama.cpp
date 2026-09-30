#!/usr/bin/env bash
# Reproducible long-context benchmark for a single GPU.
#
# For every (KV cache type, context depth) pair this reports:
#   - prefill and decode throughput measured at that depth (llama-bench)
#   - peak GPU memory while the run is live (sampled with nvidia-smi)
#   - optional perplexity on a local corpus (llama-perplexity)
#
# Every config runs REPS times and the median of the per-run averages is kept,
# so one warmup outlier cannot move the number.
#
# usage:
#   scripts/bench-longctx.sh -m model.gguf [-o outdir] [-c "8192 32768"] [-k "f16 q8_0"] [--ppl]
#
# run this from a build directory (the one holding ./bin).

set -u

BIN=${BIN:-./bin}
MODEL=
OUT=bench-longctx
CTXS="8192 32768"
KVTS="f16 q8_0"
NP=512
NG=128
REPS=3
DEV=0
DO_PPL=0
PPL_ONLY=0
CORPUS=${CORPUS:-wikitext-2-raw/wiki.test.raw}
PPL_CTX=${PPL_CTX:-512}
PPL_CHUNKS=${PPL_CHUNKS:-0}
NGL=999

usage() {
    sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'
}

while (( $# )); do
    case "$1" in
        -m) MODEL=$2; shift 2 ;;
        -o) OUT=$2; shift 2 ;;
        -c) CTXS=$2; shift 2 ;;
        -k) KVTS=$2; shift 2 ;;
        -r) REPS=$2; shift 2 ;;
        -dev) DEV=$2; shift 2 ;;
        --ppl) DO_PPL=1; shift ;;
        --ppl-only) DO_PPL=1; PPL_ONLY=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

[ -n "$MODEL" ] || { echo "missing -m <model.gguf>" >&2; usage; exit 1; }
[ -x "$BIN/llama-bench" ] || { echo "$BIN/llama-bench not found, run from a build dir" >&2; exit 1; }

mkdir -p "$OUT"
RES="$OUT/results.tsv"
printf "kv\tctx\tprefill_tps\tdecode_tps\tpeak_mib\tstatus\n" > "$RES"

# median of the numbers on stdin
median() {
    sort -n | awk '{a[NR]=$1} END{ if (NR==0) print "NA"; else if (NR%2) printf "%.2f", a[(NR+1)/2]; else printf "%.2f", (a[NR/2]+a[NR/2+1])/2 }'
}

# kv spec is "TYPE" (same for K and V) or "TYPE_K,TYPE_V"
kv_tag() { echo "$1" | tr ',' '-'; }
kv_parse() {
    case "$1" in
        *,*) echo "${1%%,*} ${1##*,}" ;;
        *)   echo "$1 $1" ;;
    esac
}

# print avg_ts of every llama-bench record whose n_gen equals $2
pick_gen() {
    python3 - "$1" "$2" <<'PY'
import json, sys
want = int(sys.argv[2])
for line in open(sys.argv[1]):
    line = line.strip()
    if not line:
        continue
    try:
        d = json.loads(line)
    except ValueError:
        continue
    if d.get("n_gen", 0) == want:
        print("%.2f" % d["avg_ts"])
PY
}

# sample GPU memory every 200ms until this subshell is killed
start_mem_sampler() {
    local file=$1
    (
        while :; do
            nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$DEV" 2>/dev/null
            sleep 0.2
        done
    ) > "$file" 2>/dev/null &
    echo $!
}

stop_mem_sampler() {
    kill "$1" 2>/dev/null
    wait "$1" 2>/dev/null
}

run_bench() {
    local kv=$1 ctx=$2
    local tag="kv$(kv_tag "$kv")_d${ctx}"
    local jsonl="$OUT/$tag.jsonl"
    local log="$OUT/$tag.log"
    local kv_k kv_v
    read -r kv_k kv_v <<< "$(kv_parse "$kv")"

    : > "$jsonl"
    local mem="$OUT/$tag.mem"; : > "$mem"
    local sampler; sampler=$(start_mem_sampler "$mem")

    local rc=0 i
    for (( i = 1; i <= REPS; i++ )); do
        "$BIN/llama-bench" \
            -m "$MODEL" -p "$NP" -n "$NG" -d "$ctx" \
            -ctk "$kv_k" -ctv "$kv_v" -fa on \
            -ngl "$NGL" -r 1 -o jsonl \
            >> "$jsonl" 2>> "$log" || { rc=$?; break; }
    done

    stop_mem_sampler "$sampler"

    local peak; peak=$(sort -n "$mem" | tail -1)

    if (( rc != 0 )); then
        printf "%s\t%s\tNA\tNA\t%s\tFAIL\n" "$kv" "$ctx" "$peak" >> "$RES"
        echo "  kv=$kv ctx=$ctx -> FAIL (see $log)"
        return
    fi

    local pp tg
    pp=$(pick_gen "$jsonl" 0 | median)
    tg=$(pick_gen "$jsonl" "$NG" | median)
    printf "%s\t%s\t%s\t%s\t%s\tOK\n" "$kv" "$ctx" "$pp" "$tg" "$peak" >> "$RES"
    echo "  kv=$kv ctx=$ctx -> prefill ${pp}t/s decode ${tg}t/s peak ${peak}MiB"
}

run_ppl() {
    local kv=$1
    local tag="kv$(kv_tag "$kv")"
    local log="$OUT/ppl_${tag}.log"
    local out="$OUT/ppl_${tag}.txt"
    local kv_k kv_v
    read -r kv_k kv_v <<< "$(kv_parse "$kv")"

    if [ ! -f "$CORPUS" ]; then
        echo "  ppl skipped: $CORPUS not found (run scripts/get-wikitext-2.sh)" >&2
        return
    fi

    local mem="$OUT/ppl_${tag}.mem"; : > "$mem"
    local sampler; sampler=$(start_mem_sampler "$mem")

    local rc=0
    "$BIN/llama-perplexity" \
        -m "$MODEL" -f "$CORPUS" -c "$PPL_CTX" \
        $( (( PPL_CHUNKS > 0 )) && echo "--chunks $PPL_CHUNKS" ) \
        -ctk "$kv_k" -ctv "$kv_v" -fa on -ngl "$NGL" \
        > "$out" 2> "$log" || rc=$?

    stop_mem_sampler "$sampler"

    if (( rc != 0 )); then
        echo "  ppl kv=$kv -> FAIL (see $log)"
        return
    fi

    local ppl peak
    # the "Final estimate" line is logged to stderr
    ppl=$(grep -oE 'Final estimate: PPL = [0-9.]+' "$log" | tail -1 | awk '{print $5}')
    peak=$(sort -n "$mem" | tail -1)
    printf "%s\t%s\t%s\n" "$kv" "${ppl:-NA}" "$peak" >> "$OUT/ppl.tsv"
    echo "  ppl kv=$kv -> ${ppl:-NA} (peak ${peak}MiB)"
}

echo "model: $MODEL"
echo "ctx:   $CTXS"
echo "kv:    $KVTS"
echo

for kv in $KVTS; do
    (( PPL_ONLY )) && break
    for ctx in $CTXS; do
        run_bench "$kv" "$ctx"
    done
done

if (( DO_PPL )); then
    printf "kv\tppl\tpeak_mib\n" > "$OUT/ppl.tsv"
    for kv in $KVTS; do
        run_ppl "$kv"
    done
fi

echo
echo "results: $RES"
if command -v column >/dev/null; then
    column -t -s $'\t' "$RES"
else
    cat "$RES"
fi
