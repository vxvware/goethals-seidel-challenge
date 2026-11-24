#!/bin/bash
# Restartable two-divisor compression census with pc2_inst.c and pc2_join.c.
#
#   usage: bash run_census2.sh PATTERN n d1 d2 [workers]   (workers default: all cores)
#   env:   OUT=dir      output directory (default out/<PATTERN>-n<n>-d<d1>x<d2>)
#          LINES=l      instance lines per join unit, rounded to whole signature
#                       groups (default 20000)
#          THREADS=t    OpenMP threads per join worker (default 1; stage 1 uses
#                       one thread per worker)
#          UNTIL=instances  stop once inst.txt and units.txt are written
#          PC_PAIR_MB, PC_CACHE_MB, PC_ORBIT_MB, PC_ORBIT_CAP, PC_MEM_MB are
#          passed to the engines; they tune memory and speed only
#
# Stages: inst writes the length-m1 instances (m1 = n/d1, one per unit orbit),
# sort orders them by their length-g signature (g = n/(d1 d2)) into inst.txt
# and inst.txt.sigs, units.txt packs whole signature groups into units of
# about LINES lines, and each join unit u writes rows-l<u>.txt.  Workers take
# units in order through a lock directory per unit; locks left by a killed run
# are removed at start, so run one driver per OUT at a time.  A finished unit leaves
# <unit>.done and one line in summary.txt; a unit with a .done marker is
# skipped.  An interrupted unit resumes after the last line in its .progress
# file (that line's signature is matched again; a quadruple can appear twice in
# the unit output, rows.txt is the sorted union).
#
# touch OUT/STOP to stop: workers stop before their next unit and a running
# join stops before its next line.  Remove the file and rerun to go on.
#
# The settings and the sha256 of both engine sources are fixed by the first
# start and checked on every restart.  The engines are built into
# OUT/bin/<hostname>.  When every unit is done, rows.txt holds the distinct
# quadruples and total.txt the counts and the summed unit CPU seconds.
set -u
PAT=${1:?pattern}; N=${2:?n}; D1=${3:?d1}; D2=${4:?d2}
if [ -n "${5:-}" ]; then W=$5; else W=$(nproc); fi
HERE=$(cd "$(dirname "$0")" && pwd) || exit 1
OUT=${OUT:-out/$PAT-n$N-d${D1}x$D2}
mkdir -p "$OUT" || exit 1
OUT=$(cd "$OUT" && pwd)
LINES=${LINES:-20000}; TH=${THREADS:-1}
CC="gcc -O3 -march=native -fopenmp"
HOST=$(uname -n); BIN=$OUT/bin/${HOST:-host}
mkdir -p "$BIN" || exit 1
build() {   # build target source
    if [ ! -x "$1" ] || [ "$2" -nt "$1" ]; then $CC "$2" -lm -o "$1.tmp.$$" && mv "$1.tmp.$$" "$1" || exit 1; fi
}
build "$BIN/pc2_inst" "$HERE/pc2_inst.c"
build "$BIN/pc2_join" "$HERE/pc2_join.c"
SRC=$(cd "$HERE" && sha256sum pc2_inst.c pc2_join.c | tr -s ' \n' '  ')
SET="pattern=$PAT n=$N d1=$D1 d2=$D2 LINES=$LINES sources: $SRC"
if [ -f "$OUT/settings.txt" ]; then
    [ "$(cat "$OUT/settings.txt")" = "$SET" ] || { echo "settings differ from $OUT/settings.txt:"; cat "$OUT/settings.txt"; exit 1; }
else
    echo "$SET" > "$OUT/settings.txt"
fi
stopped() { [ -f "$OUT/STOP" ]; }

# stage 1: instances, sorted by signature, and the units
if [ ! -f "$OUT/inst.done" ]; then
    stopped && { echo "STOP file present, stopping"; exit 3; }
    s=$(date +%s)
    "$BIN/pc2_inst" "$PAT" "$N" "$D1" "$W" "$OUT/inst.raw" > "$OUT/inst.log" 2>&1 || { echo "inst failed, see $OUT/inst.log"; exit 1; }
    "$BIN/pc2_join" sort "$PAT" "$N" "$D1" "$D2" "$OUT/inst.raw" "$OUT/inst.txt" >> "$OUT/inst.log" 2>&1 || { echo "sort failed, see $OUT/inst.log"; exit 1; }
    NI=$(wc -l < "$OUT/inst.txt")
    awk -v L="$LINES" -v N="$NI" 'BEGIN { u = 0; f = 0 } { if ($1 - f >= L) { print f, $1 - 1; f = $1 } } END { if (N > f) print f, N - 1 }' "$OUT/inst.txt.sigs" > "$OUT/units.tmp" && mv "$OUT/units.tmp" "$OUT/units.txt" || exit 1
    echo "instances $NI md5 $(md5sum < "$OUT/inst.txt" | cut -d' ' -f1) signatures $(wc -l < "$OUT/inst.txt.sigs") units $(wc -l < "$OUT/units.txt")" > "$OUT/instances.info"
    echo "inst wall=$(( $(date +%s) - s ))s $(date -Is)" >> "$OUT/summary.txt"
    touch "$OUT/inst.done"
fi
cat "$OUT/instances.info"
if [ "${UNTIL:-}" = instances ]; then exit 0; fi
NU=$(wc -l < "$OUT/units.txt")

# stage 2: join units, W workers
rmdir "$OUT"/l*.lock 2>/dev/null
worker() {
    local u=0 first last unit rc
    while read -r first last; do
        unit=l$u; u=$((u + 1))
        [ -f "$OUT/$unit.done" ] && continue
        stopped && return 3
        mkdir "$OUT/$unit.lock" 2>/dev/null || continue
        [ -f "$OUT/$unit.done" ] && { rmdir "$OUT/$unit.lock"; continue; }
        if [ -s "$OUT/$unit.progress" ]; then first=$(( $(sort -n "$OUT/$unit.progress" | tail -1) + 1 )); fi
        rc=0
        if [ "$first" -le "$last" ]; then
            ( TIMEFORMAT='%U %S'; time ( PC_STOP="$OUT/STOP" PC_PROGRESS="$OUT/$unit.progress" \
                "$BIN/pc2_join" "$PAT" "$N" "$D1" "$D2" "$OUT/inst.txt" "$TH" "$OUT/rows-$unit.txt" "$first" "$last" >> "$OUT/$unit.log" 2>&1 ) ) 2>> "$OUT/$unit.cpu"
            rc=$?
            if [ $rc -eq 0 ] && ! tail -1 "$OUT/$unit.log" | grep -q '^join complete'; then rc=1; fi
        fi
        if [ $rc -eq 0 ]; then
            touch "$OUT/rows-$unit.txt"
            echo "$unit lines=$first-$last cpu=$(awk '{ s += $1 + $2 } END { print s + 0 }' "$OUT/$unit.cpu" 2>/dev/null)s rows=$(grep -c '^[+-]' "$OUT/rows-$unit.txt") host=$HOST $(date -Is)" >> "$OUT/summary.txt"
            touch "$OUT/$unit.done"
        else
            echo "unit $unit stopped or failed (rc=$rc), see $OUT/$unit.log"
        fi
        rmdir "$OUT/$unit.lock"
        [ $rc -ne 0 ] && return $rc
    done < "$OUT/units.txt"
    return 0
}
pids=""
for w in $(seq 1 "$W"); do worker & pids="$pids $!"; sleep 1; done
fail=0
for p in $pids; do wait "$p" || fail=1; done
[ $fail -ne 0 ] && exit 3

# all units done?
u=0; files=""; missing=0
while [ $u -lt "$NU" ]; do [ -f "$OUT/l$u.done" ] || missing=1; files="$files $OUT/rows-l$u.txt"; u=$((u + 1)); done
[ $missing -ne 0 ] && { echo "units still running or unfinished; rerun when they are done"; exit 0; }
# shellcheck disable=SC2086
if [ -n "$files" ]; then LC_ALL=C sort -u $files > "$OUT/rows.txt"; else : > "$OUT/rows.txt"; fi
CPU=$(grep -o 'cpu=[0-9.]*' "$OUT/summary.txt" | cut -d= -f2 | awk '{ s += $1 } END { print s + 0 }')
{
    echo "pattern=$PAT n=$N d1=$D1 d2=$D2"
    cat "$OUT/instances.info"
    echo "join units $NU, distinct quadruples $(grep -c '^[+-]' "$OUT/rows.txt")"
    echo "join CPU seconds summed over units $CPU"
} > "$OUT/total.txt"
cat "$OUT/total.txt"
