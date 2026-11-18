#!/bin/sh
# Restartable census with best_hkt2, one unit of work per invocation of the engine.
#
#   usage: sh run_units.sh [n] [threads]        (defaults: 73, all cores)
#   env:   QB=q       2^q classes of patterns, each a unit (default 9 at n >= 65, else 0)
#          PIECES=p   also split every class into p A pieces, each (class, piece) a unit;
#                     unset: a unit is a whole class and the engine sizes its own A
#                     pieces from BEST_MEM_MB
#          OUT=dir    output directory (default out/n<n>)
#          any BEST_* knob of the engine (BEST_MEM_MB, BEST_NLEV, BEST_CCK, ...) is passed on;
#          QB, PIECES and those knobs must stay the same across restarts of one census
#
# Each unit writes <unit>.txt (solutions) and <unit>.log under OUT, then <unit>.done,
# and appends one line to OUT/summary.txt.  A unit with a .done marker is skipped, so
# the script can be killed and started again at any time: the interrupted unit has
# only a .tmp output and is run again.  When every unit is done the outputs are
# concatenated into OUT/best-<n>.txt and the counts are summed into OUT/total.txt.
# No systemd, no cgroups: wrap the script itself if limits are wanted.
set -u
N=${1:-73}
if [ -n "${2:-}" ]; then TH=$2; else TH=$(nproc); fi
cd "$(dirname "$0")" || exit 1
if [ "$N" -gt 63 ]; then BIN=./best_hkt2_wide; FLAGS=-DWIDE; QDEF=9; else BIN=./best_hkt2; FLAGS=; QDEF=0; fi
QB=${QB:-$QDEF}
OUT=${OUT:-out/n$N}
mkdir -p "$OUT" || exit 1
if [ ! -x $BIN ] || [ best_hkt2.c -nt $BIN ]; then
    gcc -O3 -march=native -fopenmp $FLAGS best_hkt2.c -lm -o $BIN || exit 1
fi
# the settings of a census are fixed by its first start
SET="n=$N QB=$QB PIECES=${PIECES:-} $(env | grep '^BEST_' | grep -v '^BEST_CLASS=\|^BEST_PIECE=\|^BEST_PIECES=\|^BEST_QB=\|^BEST_DFILE=' | sort | tr '\n' ' ')"
if [ -f "$OUT/settings.txt" ]; then
    [ "$(cat "$OUT/settings.txt")" = "$SET" ] || { echo "settings differ from $OUT/settings.txt: $(cat "$OUT/settings.txt")"; exit 1; }
else
    echo "$SET" > "$OUT/settings.txt"
fi
num() { grep -o "$1[0-9]*" "$2" | grep -o '[0-9]*$' | head -1; }
NC=$((1 << QB)); units=""; b=0
while [ $b -lt $NC ]; do
    if [ -n "${PIECES:-}" ]; then i=0; while [ $i -lt "$PIECES" ]; do units="$units c${b}_p$i"; i=$((i + 1)); done
    else units="$units c$b"; fi
    b=$((b + 1))
done
export BEST_DFILE="$OUT/dlist.bin"          # the D list is enumerated by the first unit only
for u in $units; do
    [ -f "$OUT/$u.done" ] && continue
    b=${u#c}; b=${b%%_*}
    s=$(date +%s)
    if [ -n "${PIECES:-}" ]; then
        BEST_QB=$QB BEST_CLASS=$b BEST_PIECES=$PIECES BEST_PIECE=${u##*_p} $BIN $N $TH "$OUT/$u.tmp" > "$OUT/$u.log" 2>&1
    else
        BEST_QB=$QB BEST_CLASS=$b $BIN $N $TH "$OUT/$u.tmp" > "$OUT/$u.log" 2>&1
    fi
    rc=$?; e=$(date +%s)
    if [ $rc -ne 0 ] || ! grep -q "solutions written" "$OUT/$u.log"; then echo "unit $u failed (rc=$rc), see $OUT/$u.log"; exit 1; fi
    mv "$OUT/$u.tmp" "$OUT/$u.txt" || exit 1
    echo "$u wall=$((e - s))s threads=$TH rows=$(grep -c '^[+-]' "$OUT/$u.txt") capped=$(num 'capped systems: ' "$OUT/$u.log") pairs=$(num '256-level test: ' "$OUT/$u.log") patterns=$(num 'patterns ' "$OUT/$u.log") $(date -Is)" >> "$OUT/summary.txt"
    touch "$OUT/$u.done"
done
# all units done: one output file and the totals, from the unit files themselves
: > "$OUT/best-$N.tmp"; cap=0; pairs=0; pat=0; wall=0; nu=0
for u in $units; do
    cat "$OUT/$u.txt" >> "$OUT/best-$N.tmp"
    cap=$((cap + $(num 'capped systems: ' "$OUT/$u.log"))); pairs=$((pairs + $(num '256-level test: ' "$OUT/$u.log")))
    pat=$((pat + $(num 'patterns ' "$OUT/$u.log"))); nu=$((nu + 1))
    w=$(grep "^$u wall=" "$OUT/summary.txt" | tail -1 | grep -o 'wall=[0-9]*' | grep -o '[0-9]*'); wall=$((wall + ${w:-0}))
done
mv "$OUT/best-$N.tmp" "$OUT/best-$N.txt"
echo "n=$N units=$nu rows=$(grep -c '^[+-]' "$OUT/best-$N.txt") capped=$cap pairs=$pairs patterns=$pat unit_wall_sum=${wall}s threads=$TH" | tee "$OUT/total.txt"
[ "$cap" -eq 0 ] || echo "WARNING: nullity-capped systems, the census is NOT exhaustive"
