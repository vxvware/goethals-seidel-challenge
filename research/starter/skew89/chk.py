import numpy as np
rows=[[1 if c=='+' else -1 for c in l.strip()] for l in open('sol.txt')]
n=89; assert all(len(r)==n for r in rows) and len(rows)==4
a,b,c,d=[np.array(r,dtype=np.int64) for r in rows]
paf=lambda x:[int(np.dot(x,np.roll(x,-s))) for s in range(1,n)]
tot=[sum(t) for t in zip(*map(paf,(a,b,c,d)))]
print("PAF sums all zero:",all(t==0 for t in tot), "max abs",max(map(abs,tot)))
print("A skew:", a[0]==1 and all(a[i]==-a[(n-i)%n] for i in range(1,n)))
circ=lambda x:np.array([np.roll(x,i) for i in range(n)])
A,B,C,D=map(circ,(a,b,c,d)); R=np.fliplr(np.eye(n,dtype=np.int64))
H=np.block([[A,B@R,C@R,D@R],[-B@R,A,D.T@R,-C.T@R],[-C@R,-D.T@R,A,B.T@R],[-D@R,C.T@R,-B.T@R,A]])
N=4*n
print("order",H.shape,"entries ±1:",set(np.unique(H))=={-1,1})
print("H H^T = 356 I:",np.array_equal(H@H.T,N*np.eye(N,dtype=np.int64)))
print("H + H^T = 2 I:",np.array_equal(H+H.T,2*np.eye(N,dtype=np.int64)))
