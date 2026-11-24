#!/bin/bash
# remaining stage A slices of round 3 (slice 1,0 is done); stops if STOP_R3 exists
cd ~/skew89/round3; export OMP_NUM_THREADS=16; O=r3
for s in 2,0 1,1 2,1 0,1 0,2 1,2; do
  [ -e STOP_R3 ] && break
  RED=1 SLICE=$s nice -n 19 ./tj7f b001.txt $O/A0.txt $O/A1.txt $O/A2.txt > $O/A_$s.out 2> $O/A_$s.err
  echo "$(date "+%F %T") A slice $s: $(grep -c "^solution" $O/A_$s.out) solution lines; $(grep "^join" $O/A_$s.err)" >> $O/log.txt
done
echo "$(date "+%F %T") remaining stage A slices finished" >> $O/log.txt
