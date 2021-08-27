import tensornetwork as tn
import numpy as np
import tensorly as tl
from tensorly import *
import torch
from time import *

tn.set_default_backend("pytorch")

n=120
a=n
b=n
c=n
d=n

r=int(n*0.1)

U=[0 for x in range(0,7)]
B=[0 for x in range(0,7)]
dim=[[0,1,2,3],[0,1],[2,3],0,1,2,3]
k=[1,r,r,r,r,r,r]

U[3] = torch.rand(a,k[3]).cuda()
U[4] = torch.rand(b,k[4]).cuda()
U[5] = torch.rand(c,k[5]).cuda()
U[6] = torch.rand(d,k[6]).cuda()

B[2] = torch.rand(k[5],k[6],k[2]).cuda()
B[1] = torch.rand(k[3],k[4],k[1]).cuda()
B[0] = torch.rand(k[1],k[2],k[0]).cuda()


#生成低秩矩阵
U[1] = tn.ncon([B[1],U[3],U[4]],[(1,2,-3),(-1,1),(-2,2)])
U[2] = tn.ncon([B[2],U[5],U[6]],[(1,2,-3),(-1,1),(-2,2)])
U[0] = tn.ncon([B[0],U[1],U[2]],[(1,2,-5),(-1,-2,1),(-3,-4,2)])
U[0] = torch.squeeze(U[0])


print("size:\n",n)
torch.cuda.synchronize()
begin_time = time()

#矩阵化
U[3] = U[0].permute(0,1,2,3).reshape(a,b*c*d)
U[4] = U[0].permute(1,0,2,3).reshape(b,a*c*d)
U[5] = U[0].permute(2,0,1,3).reshape(c,a*b*d)
U[6] = U[0].permute(3,0,1,2).reshape(d,a*b*c)

U[1] = U[0].permute(0,1,2,3).reshape(a*b,c*d)
U[2] = U[0].permute(2,3,0,1).reshape(c*d,a*b)

# leaf node 分解
for i in range(3,7):
    x_mat = torch.matmul(U[i],U[i].T)
    _,U_=torch.eig(x_mat,True)
    U[i]=U_[:,:k[i]]

# U[1]  U[2]
for i in range(1,3):
    U_,_,_=torch.svd(U[i])
    U[i]=U_[:,:k[i]]
#B[1]  B[2]  TTM
U[1] = U[1].reshape(a,b,k[1])
U[2] = U[2].reshape(c,d,k[2])
B[1] = tn.ncon([U[1],U[3],U[4]],[(1,2,-3),(1,-1),(2,-2)])
B[2] = tn.ncon([U[2],U[5],U[6]],[(1,2,-3),(1,-1),(2,-2)])
#U_node0 = U[0].reshape(a,b,c,d,k[0])
B[0] = tn.ncon([U[0],U[1],U[2]],[(1,2,3,4),(1,2,-1),(3,4,-2)])


torch.cuda.synchronize()
end_time = time()
run_time = end_time-begin_time
print ('cost time is：',run_time)
