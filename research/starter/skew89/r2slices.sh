#!/bin/sh
# Round-2 union join under plan w3, one top-level branch at a time (the unsliced join needs more than 26 GB and was
# killed). Each slice is a complete, disjoint part of the join; an invalid digit pair returns at once as "empty slice".
cd ~/skew89 || exit 1
log() { echo "$(date -Is) $*" >> round2.log; }
for vb in 0 1 2 3; do for vc in 0 1 2 3; do
  o=r2_slice_${vb}_${vc}
  [ -f $o.done ] && continue
  log "slice $vb,$vc started"
  SLICE=$vb,$vc nice -n 19 ./tj7s w3.txt r2_all_0.txt r2_all_1.txt r2_all_2.txt > $o.log 2> $o.err
  rc=$?
  log "slice $vb,$vc rc=$rc: $(grep -c '^solution' $o.log) solution lines; $(tail -1 $o.log | cut -c1-90)"
  [ $rc = 0 ] && touch $o.done
done; done
log "all slices finished: $(cat r2_slice_*.log | grep -c '^solution') solution lines"
touch r2_join.done
