#!/bin/bash
# Restartable compression census with pc_compress.c and pc_lift.c, one unit of
# work per engine invocation.
#
#   usage: bash run_census.sh PATTERN n d [threads]      (threads default: all cores)
#   env:   OUT=dir     output directory (default out/<PATTERN>-n<n>-d<d>)
#          PIECES=p    stage-1 pieces of the code range (default 1, 16 at n >= 63)
#          SHARDS=k    stage-2 match shards (default 1)
#          LINES=l     instance lines per lift unit (default 512)
#          WIDE=1      lift with the -DPC_WIDE build (default 1 above n = 65, else 0)
#          UNTIL=instances  stop once instances.txt is written (to lift elsewhere)
#          PC_MEM_MB, PC_HASH_LOG2, PC_PAIR_BYTES, PC_CACHE, PC_CACHE_MB are
#          passed to the engines; they tune memory and speed only
#
# Stages: enum units e<i> write lists-e<i>.txt; merge writes lists.txt; match
# units m<k> write inst-m<k>.txt; their union, sorted, is instances.txt; lift
# units l<u> lift instance lines [u LINES, (u+1) LINES) into rows-l<u>.txt.
# A finished unit leaves <unit>.done and one line in summary.txt; a unit with a
# .done marker is skipped, so the script can be killed and started again at any
# time.  An interrupted enum or match unit has only a .part file and runs again;
# an interrupted lift unit resumes after the last instance in its .progress
# file (the instance it was in is lifted again, so a quadruple can appear twice
# in the unit output; rows.txt is the sorted union).
#
# touch OUT/STOP to stop: the script stops before its next unit and a running
# lift unit stops before its next instance.  Remove the file and rerun to go on.
#
# The settings and the sha256 of both engine sources are fixed by the first
# start and checked on every restart.  The engines are built into OUT/bin/<hostname>.  When every unit is done, rows.txt holds
# the distinct quadruples and total.txt the counts and wall-clock sums.
set -u
PAT=${1:?pattern}; N=${2:?n}; D=${3:?d}
if [ -n "${4:-}" ]; then TH=$4; else TH=$(nproc); fi
HERE=$(cd "$(dirname "$0")" && pwd) || exit 1
OUT=${OUT:-out/$PAT-n$N-d$D}
mkdir -p "$OUT" || exit 1
OUT=$(cd "$OUT" && pwd)
if [ "$N" -ge 63 ]; then PDEF=16; else PDEF=1; fi
PIECES=${PIECES:-$PDEF}; SHARDS=${SHARDS:-1}; LINES=${LINES:-512}
if [ "$N" -gt 65 ]; then WDEF=1; else WDEF=0; fi
WIDE=${WIDE:-$WDEF}
CC="gcc -O3 -march=native -fopenmp"
build() {   # build target source flags
    if [ ! -x "$1" ] || [ "$2" -nt "$1" ]; then $CC $3 "$2" -lm -o "$1.tmp" && mv "$1.tmp" "$1" || exit 1; fi
}
HOST=$(uname -n); BIN=$OUT/bin/${HOST:-host}   # -march=native binaries belong to one host
mkdir -p "$BIN" || exit 1
COMP=$BIN/pc_compress; build "$COMP" "$HERE/pc_compress.c" ""
if [ "$WIDE" = 1 ]; then LIFT=$BIN/pc_lift_wide; build "$LIFT" "$HERE/pc_lift.c" -DPC_WIDE
else LIFT=$BIN/pc_lift; build "$LIFT" "$HERE/pc_lift.c" ""; fi
SRC=$(cd "$HERE" && sha256sum pc_compress.c pc_lift.c | tr -s ' \n' '  ')
SET="pattern=$PAT n=$N d=$D PIECES=$PIECES SHARDS=$SHARDS LINES=$LINES WIDE=$WIDE sources: $SRC"
if [ -f "$OUT/settings.txt" ]; then
    [ "$(cat "$OUT/settings.txt")" = "$SET" ] || { echo "settings differ from $OUT/settings.txt:"; cat "$OUT/settings.txt"; exit 1; }
else
    echo "$SET" > "$OUT/settings.txt"
fi
stopped() { if [ -f "$OUT/STOP" ]; then echo "STOP file present, stopping"; exit 3; fi; }
record() {  # unit start-seconds extra; a resumed lift unit adds the wall of its earlier runs
    prev=0; [ -f "$OUT/$1.wall" ] && prev=$(awk '{ s += $1 } END { print s + 0 }' "$OUT/$1.wall")
    echo "$1 wall=$(( $(date +%s) - $2 + prev ))s threads=$TH $3 $(date -Is)" >> "$OUT/summary.txt"
    touch "$OUT/$1.done"
}

# stage 1: enumeration pieces, then the merged list file
i=0; lists=""
while [ $i -lt "$PIECES" ]; do
    u=e$i; lists="$lists $OUT/lists-$u.txt"
    if [ ! -f "$OUT/$u.done" ]; then
        stopped; s=$(date +%s)
        "$COMP" enum "$PAT" "$N" "$D" "$TH" "$OUT/lists-$u.txt" $i "$PIECES" > "$OUT/$u.log" 2>&1
        rc=$?
        if [ $rc -ne 0 ] || [ ! -f "$OUT/lists-$u.txt" ]; then echo "unit $u failed (rc=$rc), see $OUT/$u.log"; exit 1; fi
        record $u "$s" "$(grep -o 'skew .*survive' "$OUT/$u.log")"
    fi
    i=$((i + 1))
done
if [ ! -f "$OUT/merge.done" ]; then
    stopped; s=$(date +%s)
    # shellcheck disable=SC2086
    "$COMP" merge "$OUT/lists.txt" $lists > "$OUT/merge.log" 2>&1 || { echo "merge failed, see $OUT/merge.log"; exit 1; }
    record merge "$s" "$(cat "$OUT/merge.log")"
fi

# stage 2: match shards, then the sorted instance file
k=0
while [ $k -lt "$SHARDS" ]; do
    u=m$k
    if [ ! -f "$OUT/$u.done" ]; then
        stopped; s=$(date +%s)
        "$COMP" match "$OUT/lists.txt" "$TH" "$OUT/inst-$u.txt" $k "$SHARDS" > "$OUT/$u.log" 2>&1
        rc=$?
        if [ $rc -ne 0 ] || [ ! -f "$OUT/inst-$u.txt" ]; then echo "unit $u failed (rc=$rc), see $OUT/$u.log"; exit 1; fi
        record $u "$s" "$(grep '^total:' "$OUT/$u.log")"
    fi
    k=$((k + 1))
done
if [ ! -f "$OUT/instances.done" ]; then
    k=0; files=""
    while [ $k -lt "$SHARDS" ]; do files="$files $OUT/inst-m$k.txt"; k=$((k + 1)); done
    # shellcheck disable=SC2086
    LC_ALL=C sort $files > "$OUT/instances.tmp" && mv "$OUT/instances.tmp" "$OUT/instances.txt" || exit 1
    echo "instances $(wc -l < "$OUT/instances.txt") md5 $(md5sum < "$OUT/instances.txt" | cut -d' ' -f1)" > "$OUT/instances.info"
    touch "$OUT/instances.done"
fi
NI=$(wc -l < "$OUT/instances.txt")
if [ "${UNTIL:-}" = instances ]; then cat "$OUT/instances.info"; exit 0; fi

# stage 3: lift units
NU=$(( (NI + LINES - 1) / LINES ))
u=0
while [ $u -lt $NU ]; do
    unit=l$u
    if [ ! -f "$OUT/$unit.done" ]; then
        stopped; s=$(date +%s)
        first=$((u * LINES)); last=$((first + LINES - 1))
        if [ -s "$OUT/$unit.progress" ]; then first=$(( $(sort -n "$OUT/$unit.progress" | tail -1) + 1 )); fi
        if [ $first -le $last ]; then
            PC_STOP="$OUT/STOP" PC_PROGRESS="$OUT/$unit.progress" \
                "$LIFT" "$N" "$D" "$OUT/instances.txt" "$TH" "$OUT/rows-$unit.txt" $first $last >> "$OUT/$unit.log" 2>&1
            rc=$?
            if [ $rc -ne 0 ]; then echo $(( $(date +%s) - s )) >> "$OUT/$unit.wall"; fi
            if [ $rc -eq 3 ]; then echo "stopped inside unit $unit; rerun to resume"; exit 3; fi
            if [ $rc -ne 0 ] || ! tail -1 "$OUT/$unit.log" | grep -q '^lift complete'; then echo "unit $unit failed (rc=$rc), see $OUT/$unit.log"; exit 1; fi
        fi
        touch "$OUT/rows-$unit.txt"
        record $unit "$s" "lines=$((u * LINES))-$last rows=$(grep -c '^[+-]' "$OUT/rows-$unit.txt")"
    fi
    u=$((u + 1))
done

# all units done
u=0; files=""
while [ $u -lt $NU ]; do files="$files $OUT/rows-l$u.txt"; u=$((u + 1)); done
# shellcheck disable=SC2086
if [ -n "$files" ]; then LC_ALL=C sort -u $files > "$OUT/rows.txt"; else : > "$OUT/rows.txt"; fi
WALL=$(grep -o 'wall=[0-9]*' "$OUT/summary.txt" | cut -d= -f2 | awk '{ s += $1 } END { print s + 0 }')
{
    echo "pattern=$PAT n=$N d=$D"
    cat "$OUT/instances.info"
    echo "lift units $NU of $LINES lines, distinct quadruples $(grep -c '^[+-]' "$OUT/rows.txt")"
    echo "wall seconds summed over units $WALL at $TH threads"
} > "$OUT/total.txt"
cat "$OUT/total.txt"
