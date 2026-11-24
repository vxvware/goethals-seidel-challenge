#!/bin/bash
# Restartable HKT census with hkt_census.c, one unit of work per engine
# invocation.
#
#   usage: bash run_census.sh wil|good n [threads]      (threads default: all cores)
#   env:   OUT=dir      output directory (default out/<family>-n<n>)
#          MEM_MB=m     memory budget of one unit in MB (default 20000); the plan
#                       splits every group into A parts and B pieces under it
#          UNIT_PAIRS=q at most q estimated pairs per unit (default 4e11); the
#                       plan cuts every (A part, B piece) into slice ranges
#          APARTS=j BPIECES=p SLICES=s   use these splits in every group instead
#                       of the plan's
#          HKT_* other than the unit and plan variables are passed to the
#          engine; they tune speed and memory only
#
# The first start runs the engine's plan mode (HKT_PLAN=1) and keeps its PLAN
# lines in plan.txt; each group g is then cut into units g<g>-a<j>-b<i>-s<l>,
# B piece i (a range of B codes) against A part j (every J-th chunk of A
# codes) over slice range l, which partition the pairs of the group.  A unit writes rows-<unit>.txt and
# <unit>.log; a finished unit leaves <unit>.done and one line in summary.txt,
# and a unit with a .done marker is skipped, so the script can be killed and
# started again at any time.  An interrupted unit runs again from its start.
#
# touch OUT/STOP to stop: a running unit stops within 64 slices (exit 3,
# nothing written) and the script stops before its next unit.  Remove the file
# and rerun to go on.
#
# The settings and the sha256 of the engine sources are fixed by the first
# start and checked on every restart.  The engine is built into
# OUT/bin/<hostname>.  When every unit is done, rows.txt holds the distinct
# quadruples (key order, one line each) and total.txt the pair count, the
# nullity-capped count (the census is exhaustive only when it is 0) and the
# wall-clock sum.
set -u
FAM=${1:?wil or good}; N=${2:?n}
case "$FAM" in wil|good) ;; *) echo "family must be wil or good"; exit 1 ;; esac
if [ -n "${3:-}" ]; then TH=$3; else TH=$(nproc); fi
HERE=$(cd "$(dirname "$0")" && pwd) || exit 1
OUT=${OUT:-out/$FAM-n$N}
mkdir -p "$OUT" || exit 1
OUT=$(cd "$OUT" && pwd)
MEM_MB=${MEM_MB:-20000}
HOST=$(uname -n); BIN=$OUT/bin/${HOST:-host}   # -march=native binaries belong to one host
mkdir -p "$BIN" || exit 1
ENG=$BIN/hkt_census
SRCS="$HERE/hkt_census.c $HERE/../kkss/hkt_solve64.h"
if [ ! -x "$ENG" ] || [ "$HERE/hkt_census.c" -nt "$ENG" ] || [ "$HERE/../kkss/hkt_solve64.h" -nt "$ENG" ]; then
    gcc -O3 -march=native -fopenmp -DHKT_WIDE -I "$HERE/../kkss" "$HERE/hkt_census.c" -lm -o "$ENG.tmp" && mv "$ENG.tmp" "$ENG" || exit 1
fi
# shellcheck disable=SC2086
SRC=$(sha256sum $SRCS | awk '{ printf "%s %s  ", $1, $2 }' | sed "s#$HERE/##g")
UNIT_PAIRS=${UNIT_PAIRS:-4e11}
SET="family=$FAM n=$N MEM_MB=$MEM_MB UNIT_PAIRS=$UNIT_PAIRS APARTS=${APARTS:-plan} BPIECES=${BPIECES:-plan} SLICES=${SLICES:-plan} sources: $SRC"
if [ -f "$OUT/settings.txt" ]; then
    [ "$(cat "$OUT/settings.txt")" = "$SET" ] || { echo "settings differ from $OUT/settings.txt:"; cat "$OUT/settings.txt"; exit 1; }
else
    echo "$SET" > "$OUT/settings.txt"
fi
stopped() { if [ -f "$OUT/STOP" ]; then echo "STOP file present, stopping"; exit 3; fi; }
# the unit and plan variables belong to this script
unset HKT_DONLY HKT_BPIECES HKT_BPIECE HKT_APARTS HKT_APART HKT_SLICES HKT_SLICE HKT_UNIT_PAIRS HKT_PLAN HKT_SAMPLE HKT_ASAMPLE HKT_REPEAT HKT_ENUM_ONLY HKT_LIST HKT_CHECK_ONLY

if [ ! -f "$OUT/plan.txt" ]; then
    stopped
    HKT_PLAN=1 HKT_MEM_MB=$MEM_MB HKT_UNIT_PAIRS=$UNIT_PAIRS "$ENG" "$FAM" "$N" "$TH" > "$OUT/plan.log" 2>&1 || { echo "plan failed, see $OUT/plan.log"; exit 1; }
    grep '^PLAN' "$OUT/plan.log" > "$OUT/plan.tmp" && mv "$OUT/plan.tmp" "$OUT/plan.txt" || exit 1
fi
field() { echo "$1" | tr ' ' '\n' | grep "^$2=" | cut -d= -f2; }

units=""
while read -r line; do
    g=$(field "$line" group); [ -n "$g" ] || continue
    J=${APARTS:-$(field "$line" aparts)}; P=${BPIECES:-$(field "$line" bpieces)}; S=${SLICES:-$(field "$line" slices)}
    j=0
    while [ $j -lt "$J" ]; do
        i=0
        while [ $i -lt "$P" ]; do
            l=0
            while [ $l -lt "$S" ]; do units="$units g$g-a$j.$J-b$i.$P-s$l.$S"; l=$((l + 1)); done
            i=$((i + 1))
        done
        j=$((j + 1))
    done
done < "$OUT/plan.txt"
NU=$(echo $units | wc -w)
echo "$NU units" > "$OUT/units.txt"

for u in $units; do
    [ -f "$OUT/$u.done" ] && continue
    stopped
    g=${u#g}; g=${g%%-*}
    a=${u#*-a}; a=${a%%-*}; j=${a%.*}; J=${a#*.}
    b=${u#*-b}; b=${b%%-*}; i=${b%.*}; P=${b#*.}
    c=${u##*-s}; l=${c%.*}; S=${c#*.}
    s=$(date +%s)
    HKT_DONLY=$g HKT_APARTS=$J HKT_APART=$j HKT_BPIECES=$P HKT_BPIECE=$i HKT_SLICES=$S HKT_SLICE=$l HKT_STOP="$OUT/STOP" \
        "$ENG" "$FAM" "$N" "$TH" "$OUT/rows-$u.txt" > "$OUT/$u.log" 2>&1
    rc=$?
    if [ $rc -eq 3 ]; then echo "stopped inside unit $u; rerun to start it again"; exit 3; fi
    if [ $rc -ne 0 ] || ! grep -q '^UNIT' "$OUT/$u.log" || [ ! -f "$OUT/rows-$u.txt" ]; then
        echo "unit $u failed (rc=$rc), see $OUT/$u.log"; exit 1
    fi
    echo "$u wall=$(( $(date +%s) - s ))s threads=$TH $(grep '^UNIT' "$OUT/$u.log") $(date -Is)" >> "$OUT/summary.txt"
    touch "$OUT/$u.done"
done

files=""
for u in $units; do files="$files $OUT/rows-$u.txt"; done
# shellcheck disable=SC2086
LC_ALL=C sort -u $files > "$OUT/rows.tmp" && mv "$OUT/rows.tmp" "$OUT/rows.txt" || exit 1
sum() { grep -o " $1=[0-9]*" "$OUT/summary.txt" | cut -d= -f2 | awk '{ s += $1 } END { print s + 0 }'; }
{
    echo "family=$FAM n=$N units=$NU"
    echo "pairs $(sum pairs), nullity-capped solves $(sum capped)"
    echo "distinct quadruples $(grep -c '^[+-]' "$OUT/rows.txt")"
    echo "wall seconds summed over units $(grep -o 'wall=[0-9]*' "$OUT/summary.txt" | cut -d= -f2 | awk '{ s += $1 } END { print s + 0 }') at $TH threads"
    [ "$(sum capped)" = 0 ] || echo "WARNING: solves were dropped by the nullity cap; the census is not exhaustive"
} > "$OUT/total.txt"
cat "$OUT/total.txt"
