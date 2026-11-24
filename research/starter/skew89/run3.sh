#!/bin/bash
# run3.sh: round 3 of the skew 356 search (n = 89) on one machine. Stage A joins the three core pools (n2 windows for blocks 0 and
# 2, b001 windows for block 1); stage B, run only if stage A finds nothing, adds the block-1 classes outside the core.
# Needs in the working directory: pgu.c, dedup.c, tj7f.c, n2.txt, b001.txt. Set R2_0, R2_1, R2_2 to the round-2 hit files of blocks
# 0, 1, 2 (rows 3, 11, 15); each may be a space-separated list. Optional: THREADS (16), SEED (8930000; the four generator runs use
# SEED+1 .. SEED+4, which must not repeat seeds of earlier runs), SCALE (1; multiplies every hit target, for a dry run),
# OUT (r3), STOP_AT_FIRST (1: stop after the first slice that prints a solution), ONEJOIN (0; 1 skips the stage-A join and joins
# everything once under n2.txt: about 4 hours less in total, no result before the single join starts).
set -u
: "${R2_0:?set R2_0 R2_1 R2_2 to the round-2 hit files}" "${R2_1:?}" "${R2_2:?}"
T=${THREADS:-16}; ONE=${ONEJOIN:-0}; SC=${SCALE:-1}; SEED=${SEED:-8930000}; O=${OUT:-r3}; STOP=${STOP_AT_FIRST:-1}; export OMP_NUM_THREADS=$T
sc() { awk -v a="$1" -v s="$SC" -v lo="${2:-1}" 'BEGIN { v = a * s; if (v < lo) v = lo; printf "%d", v }'; }
for f in pgu dedup tj7f; do [ -x $f ] || gcc -O3 -march=native -fopenmp -o $f $f.c -lm || exit 1; done
mkdir -p $O; log() { echo "$(date '+%F %T') $*" | tee -a $O/log.txt; }
slices() {   # slices PLAN TAG file0 file1 file2 : slices 1,0 and 2,0 hold 96% of the work and 99% of block 1; the rest follow
  local plan=$1 tag=$2 found=0; shift 2
  for s in 1,0 2,0 1,1 2,1 0,1 0,2 1,2; do
    RED=1 SLICE=$s ./tj7f $plan "$@" > $O/${tag}_$s.out 2> $O/${tag}_$s.err; local c; c=$(grep -c '^solution' $O/${tag}_$s.out)
    log "$tag slice $s: $c solution lines; $(grep '^join' $O/${tag}_$s.err); $(grep 'peak' $O/${tag}_$s.err | tr -s ' \t' ' ')"
    found=$((found + c)); if [ "$c" -gt 0 ] && [ "$STOP" = 1 ]; then break; fi
  done
  log "$tag: $found solution lines in total"; [ "$found" -gt 0 ]
}
# ---- stage A: core hits (targets at SCALE = 1: 1.25e8, 8.7e7, 7.9e7; about 3.4, 2.3 and 1.8 hours at 10,300, 10,500 and 12,000 hits/s)
log "stage A generation"
timeout 16000 ./pgu n2.txt   0 $O/core_0.txt 0 $((SEED + 1)) $(sc 125000000) >> $O/gen.log 2>&1
timeout 11000 ./pgu b001.txt 1 $O/core_1.txt 0 $((SEED + 2)) $(sc 87000000)  >> $O/gen.log 2>&1
timeout 9000  ./pgu n2.txt   2 $O/core_2.txt 0 $((SEED + 3)) $(sc 79000000)  >> $O/gen.log 2>&1
log "stage A dedup"
cat $R2_0 $O/core_0.txt | ./dedup n2.txt   0 $(sc 100000000 2000000) > $O/A0.txt 2> $O/A0.dedup
cat $O/core_1.txt $R2_1 | ./dedup b001.txt 1 $(sc 100000000 2000000) $O/rest_1.txt > $O/A1.txt 2> $O/A1.dedup
cat $R2_2 $O/core_2.txt | ./dedup n2.txt   2 $(sc 100000000 2000000) > $O/A2.txt 2> $O/A2.dedup
log "classes: $(cat $O/A0.dedup | head -1) | $(cat $O/A1.dedup | head -1) | $(cat $O/A2.dedup | head -1)"
if [ "$ONE" != 1 ]; then
  log "stage A join (b001.txt)"
  if slices b001.txt A $O/A0.txt $O/A1.txt $O/A2.txt; then log "solutions found in stage A; stage B not run"; exit 0; fi
fi
# ---- stage B: block-1 classes outside the core, up to the 32-bit id limit of tj7 (3.9e8 rows per block, the stage-A classes included)
log "stage B generation"
timeout 7200 ./pgu n2.txt 1 $O/n2_1.txt 0 $((SEED + 4)) $(sc 460000000) >> $O/gen.log 2>&1
nA=$(wc -l < $O/A1.txt)
if [ "$ONE" = 1 ]; then nA=0; fi          # single join: the stage-A classes of block 1 stay in the pool, first in the file
cat $O/A1.txt $O/rest_1.txt $O/n2_1.txt | ./dedup n2.txt 1 $(sc 390000000 8000000) 2> $O/B1.dedup | tail -n +$((nA + 1)) > $O/B1.txt
log "stage B block 1: $(head -1 $O/B1.dedup); $(wc -l < $O/B1.txt) classes in the stage-B pool"
log "stage B join (n2.txt)"
slices n2.txt B $O/A0.txt $O/B1.txt $O/A2.txt; log "done"
