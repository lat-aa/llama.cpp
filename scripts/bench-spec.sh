#!/usr/bin/env bash
# Measure speculative-decoding decode speed and acceptance rate.
#
# llama-bench cannot drive speculative decoding, so this uses the
# llama-speculative-simple example, which reports both the decode speed and the
# acceptance statistics. Each config is repeated REPS times and the median
# decode speed is kept.
#
# The prompt comes from a file so runs are reproducible and free of shell
# quoting issues. Use a long prompt to probe a given KV depth: the benefit of
# speculative decoding grows with depth because each KV read is amortised over
# the accepted tokens.
#
# usage:
#   scripts/bench-spec.sh -m model.gguf -f prompt.txt [-t "none ngram-simple"] [-c 8192] [-n 128]
#
# run from a build directory (the one holding ./bin).

set -u

BIN=${BIN:-./bin}
MODEL=
PROMPT=
OUT=bench-spec
TOOL=cli
TYPES="none ngram-simple ngram-mod ngram-map-k"
CTX=8192
NP=128
REPS=3
SEED=42
DEV=0
NGL=999
CTK=f16
CTV=f16
UBATCH=512
BATCH=""
EXTRA=""

usage() {
    sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'
}

while (( $# )); do
    case "$1" in
        -m) MODEL=$2; shift 2 ;;
        -f) PROMPT=$2; shift 2 ;;
        -o) OUT=$2; shift 2 ;;
        -t) TYPES=$2; shift 2 ;;
        --tool) TOOL=$2; shift 2 ;;
        -c) CTX=$2; shift 2 ;;
        -n) NP=$2; shift 2 ;;
        -b) BATCH=$2; shift 2 ;;
        -ub) UBATCH=$2; shift 2 ;;
        -r) REPS=$2; shift 2 ;;
        -k) CTK=$2; CTV=$2; shift 2 ;;
        -x) EXTRA=$2; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

[ -n "$MODEL" ] || { echo "missing -m <model.gguf>" >&2; usage; exit 1; }
[ -f "$PROMPT" ] || { echo "missing or unreadable -f <prompt.txt>" >&2; usage; exit 1; }
if [ "$TOOL" = "spec" ]; then
    [ -x "$BIN/llama-speculative-simple" ] || { echo "$BIN/llama-speculative-simple not found, run from a build dir" >&2; exit 1; }
else
    [ -x "$BIN/llama-cli" ] || { echo "$BIN/llama-cli not found, run from a build dir" >&2; exit 1; }
fi

# the example submits the whole prompt as one batch, so the logical batch must
# cover it; the physical ubatch keeps the compute buffer small
[ -n "$BATCH" ] || BATCH=$CTX

mkdir -p "$OUT"
RES="$OUT/results.tsv"
printf "spec\tctx\tdecode_tps\taccept_pct\tn_drafted\tn_accept\tstatus\n" > "$RES"

median() {
    sort -n | awk '{a[NR]=$1} END{ if (NR==0) print "NA"; else if (NR%2) printf "%.2f", a[(NR+1)/2]; else printf "%.2f", (a[NR/2]+a[NR/2+1])/2 }'
}

# one field out of a run log
field() {
    sed -n "$2" "$1" | tail -1
}

run_type() {
    local t=$1
    local tag="spec${t}_c${CTX}"
    local log="$OUT/$tag.log"
    : > "$log"

    local speeds="$OUT/$tag.speeds"; : > "$speeds"
    local acc="" nd="" na=""

    local rc=0 i
    for (( i = 1; i <= REPS; i++ )); do
        if [ "$TOOL" = "spec" ]; then
            "$BIN/llama-speculative-simple" \
                -m "$MODEL" -f "$PROMPT" -n "$NP" -c "$CTX" \
                -b "$BATCH" -ub "$UBATCH" \
                --seed "$SEED" --temp 0 \
                --spec-type "$t" \
                -ngl "$NGL" -fa on -ctk "$CTK" -ctv "$CTV" \
                $EXTRA \
                >> "$log" 2>&1 || { rc=$?; break; }
        else
            # llama-cli applies the chat template, so the model actually follows
            # the prompt instead of continuing it verbatim - the realistic path
            "$BIN/llama-cli" \
                -m "$MODEL" -f "$PROMPT" -n "$NP" -c "$CTX" \
                -b "$BATCH" -ub "$UBATCH" \
                --seed "$SEED" --temp 0 \
                --spec-type "$t" \
                -ngl "$NGL" -fa on -ctk "$CTK" -ctv "$CTV" \
                -st $EXTRA \
                >> "$log" 2>&1 || { rc=$?; break; }
        fi

        local v
        if [ "$TOOL" = "spec" ]; then
            v=$(field "$log" 's/.*decoded .*speed: *\([0-9.]*\) t\/s.*/\1/p')
        else
            v=$(field "$log" 's/.*Generation: *\([0-9.]*\) t\/s.*/\1/p')
        fi
        [ -n "$v" ] && echo "$v" >> "$speeds"
    done

    # acceptance stats come from the last run (only the spec tool reports them)
    if [ "$TOOL" = "spec" ]; then
        acc=$(field "$log" 's/.*accept *= *\([0-9.]*\)%.*/\1/p')
        nd=$(field "$log"  's/.*n_drafted *= *\([0-9]*\).*/\1/p')
        na=$(field "$log"  's/.*n_accept *= *\([0-9]*\).*/\1/p')
    fi

    local dec; dec=$(median < "$speeds")
    if (( rc != 0 )) || [ "$dec" = "NA" ]; then
        printf "%s\t%s\tNA\tNA\tNA\tNA\tFAIL\n" "$t" "$CTX" >> "$RES"
        echo "  spec=$t ctx=$CTX -> FAIL (see $log)"
        return
    fi
    printf "%s\t%s\t%s\t%s\t%s\t%s\tOK\n" "$t" "$CTX" "$dec" "${acc:-NA}" "${nd:-NA}" "${na:-NA}" >> "$RES"
    echo "  spec=$t ctx=$CTX -> decode ${dec}t/s  accept ${acc:-NA}%  (${na:-NA}/${nd:-NA})"
}

echo "model:  $MODEL"
echo "prompt: $PROMPT ($(wc -c < "$PROMPT") bytes)"
echo "ctx:    $CTX"
echo "spec:   $TYPES"
echo

for t in $TYPES; do
    run_type "$t"
done

echo
echo "results: $RES"
if command -v column >/dev/null; then
    column -t -s $'\t' "$RES"
else
    cat "$RES"
fi
