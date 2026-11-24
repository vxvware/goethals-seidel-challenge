#!/bin/sh
# Pool generation for the n=89 plan n2 (skew block invariant under the order-11 multipliers; one lag class
# widened to four values on block 1 only). Rotates through the three blocks, two hours each; hit files are append-only, so it
# can be stopped at any time. Targets over all hosts: about 1.7e7 hits for blocks 0 and 2, 1.1e8 for block 1; merge host files with cat.
cd "$(dirname "$0")" || exit 1
host=$(uname -n); hs=$(uname -n | cksum | cut -d" " -f1); i=0
while :; do
  for b in 0 1 2; do
    i=$((i+1))
    if [ $b = 1 ]; then G="env EP=1 KR=30000 ./pg4"; T=0.45; else G="env KR=300000 ./pg2"; T=0.7; fi; timeout 7200 nice -n 19 $G n2.txt $b n2_hits_${b}_$host.txt $T $(( $(date +%s) + i + hs )) 2000000000 >> n2_gen.log 2>&1
    echo "$(date -Is) block $b file has $(wc -l < n2_hits_${b}_$host.txt) hits" >> n2_gen.log
  done
done
