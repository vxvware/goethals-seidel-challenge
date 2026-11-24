#!/bin/sh
# Second round for n=89: new hits under plan w3 (its windows contain n2's, block 1 identical), to be joined as one
# union with every n2 hit under w3.txt. Time shares follow the 600 thread-hour allocation: blocks 0 and 2 get
# 50 and 60 minutes per cycle, block 1 six. Append-only; stop at any time.
cd "$(dirname "$0")" || exit 1
host=$(uname -n); hs=$(uname -n | cksum | cut -d" " -f1); i=0
while :; do
  i=$((i+1))
  KR=300000 timeout 3000 nice -n 19 ./pg2 w3.txt 0 w3_hits_0_$host.txt 0.7 $(( $(date +%s) + i + hs )) 2000000000 >> w3b_gen.log 2>&1
  KR=300000 timeout 3600 nice -n 19 ./pg2 w3.txt 2 w3_hits_2_$host.txt 0.7 $(( $(date +%s) + i + hs + 3 )) 2000000000 >> w3b_gen.log 2>&1
  EP=1 KR=30000 timeout 360 nice -n 19 ./pg4 n2.txt 1 n2b_hits_1_$host.txt 0.45 $(( $(date +%s) + i + hs + 5 )) 2000000000 >> w3b_gen.log 2>&1
done
