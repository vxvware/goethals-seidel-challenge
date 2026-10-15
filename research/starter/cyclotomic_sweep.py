#!/usr/bin/env python3
"""Bounded, exact-PAF coset-union search. No rounded PSD hash can lose solutions."""
import argparse
import json
from math import isqrt
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from verify.gs_verify import paf,validate_rows


def cosets(p,d):
    if p<3 or any(p%q==0 for q in range(2,isqrt(p)+1)) or d%2!=1 or (p-1)%d:
        raise ValueError('requires a prime p and odd subgroup order d dividing p-1')
    subgroup={x for x in range(1,p) if pow(x,d,p)==1}
    seen=set();out=[]
    for x in range(1,p):
        if x in seen:continue
        coset={x*h%p for h in subgroup};out.append(coset);seen|=coset
    if len(out)>12:raise ValueError('starter search capped at 12 cosets')
    return out


def search(p,d):
    sets=cosets(p,d);free=[];skew=[]
    for mask in range(1<<len(sets)):
        block=set().union(*(sets[i] for i in range(len(sets)) if mask&(1<<i)))
        for zero in (False,True):
            b=block|({0} if zero else set())
            row=tuple(-1 if j in b else 1 for j in range(p))
            key=paf(row)[1:p//2+1]
            free.append((row,key,sum(row)**2))
            if not zero and all(row[j]==-row[-j] for j in range(1,p)):
                skew.append((row,key))
    index={}
    for row,key,square in free:index.setdefault(key,[]).append((row,square))
    for a,pa in skew:
        for b,pb,sb in free:
            for c,pc,sc in free:
                needed=4*p-1-sb-sc
                if needed<0 or isqrt(needed)**2!=needed:continue
                key=tuple(-x-y-z for x,y,z in zip(pa,pb,pc))
                for drow,sd in index.get(key,[]):
                    if sd!=needed:continue
                    q=(a,b,c,drow);validate_rows(q);yield q


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('prime',type=int);p.add_argument('--subgroup',type=int,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    count=0
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as stream:
        for q in search(a.prime,a.subgroup):
            stream.write(json.dumps({'n':a.prime,'rows':q})+'\n');count+=1
    print(json.dumps({'solutions':count,'scope':{'p':a.prime,'subgroup_order':a.subgroup},'output':str(a.output)}))
