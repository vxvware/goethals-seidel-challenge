#!/bin/sh
# Round 2 on K11 with the fast generator pgu: bring the new-row files to their quotas, then join the union of every
# n2 and w3 row under plan w3 with tj7b. Quotas are the 600 thread-hour allocation: 3.1e7 new rows for block 0,
# 4.3e7 for block 2 (both under w3's wider windows), 3.6e6 for block 1 (same windows in n2 and w3).
cd ~/skew89 || exit 1
log() { echo "$(date -Is) $*" >> round2.log; }
gen() { # plan block file quota
  have=$(( $(stat -c %s "$3" 2>/dev/null || echo 0) / 90 )); want=$(( $4 - have ))
  [ $want -le 0 ] && { log "block $2: $have rows, quota met"; return; }
  log "block $2: $have rows, generating $want"
  nice -n 19 ./pgu "$1" "$2" "$3" 0 $(( $(date +%s) * 7 + $2 )) $want >> round2_gen.log 2>&1
  truncate -s $(( $(stat -c %s "$3") / 90 * 90 )) "$3"
  log "block $2: now $(( $(stat -c %s "$3") / 90 )) rows"
}
gen w3.txt 0 w3_hits_0_k11.txt 31000000
gen w3.txt 2 w3_hits_2_k11.txt 43000000
gen n2.txt 1 n2b_hits_1_k11.txt 3600000
for b in 0 1 2; do cat n2_all_$b.txt w3_hits_${b}_k11.txt n2b_hits_${b}_k11.txt 2>/dev/null > r2_all_$b.txt; log "pool $b: $(( $(stat -c %s r2_all_$b.txt) / 90 )) rows"; done
log "join started"
nice -n 19 ./tj7b w3.txt r2_all_0.txt r2_all_1.txt r2_all_2.txt > r2_join.log 2> r2_join.err
log "join finished: $(grep -c '^solution' r2_join.log) solution lines"
touch r2_join.done
