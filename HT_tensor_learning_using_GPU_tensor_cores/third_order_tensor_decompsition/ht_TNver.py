import tensornetwork as tn
import numpy as np
import tensorly as tl
from tensorly import *
import torch
from time import *

tn.set_default_backend("pytorch")
n=560
a=n
b=n
c=n
r=int(n*0.5)
U=[0 for x in range(0,5)]
B=[0 for x in range(0,5)]
k=[1,r,r,r,r]


#随机数组
U[4] = torch.rand(b,k[4]).cuda()
U[3] = torch.rand(a,k[3]).cuda()
U[2] = torch.rand(c,k[2]).cuda()
B[1] = torch.rand(k[3],k[4],k[1]).cuda()
B[0] = torch.rand(k[1],k[2],k[0]).cuda()

#生成低秩矩阵
U[1] = tn.ncon([B[1],U[3],U[4]],[(1,2,-3),(-1,1),(-2,2)])
U[0] = tn.ncon([B[0],U[1],U[2]],[(1,2,-4),(-1,-2,1),(-3,2)])

U[0] = torch.squeeze(U[0]) #低秩矩阵

print("size:\n",n)
torch.cuda.synchronize()
begin_time = time()

#矩阵化
U[3] = U[0].permute(0,1,2).reshape(a,b*c)
U[4] = U[0].permute(1,0,2).reshape(b,a*c)
U[2] = U[0].permute(2,0,1).reshape(c,a*b)
U[1] = U[0].permute(0,1,2).reshape(a*b,c)


U_node=[0 for x in range(0,5)]
#U_node[0] = tn.Node(U[0].reshape(a,b,c,1))
U_node[0] = U[0].reshape(a,b,c,1)
#svd分解部分
for i in range(4,0,-1):
    #U_node[i] = tn.Node(U[i])
    if i>1:
    	x_mat = torch.matmul(U[i],U[i].T)
    	_,U_=torch.eig(x_mat,True)
    	U_node[i]=U_[:,:k[i]]
    else:
    	#U_node[i],_,_,_=torch.svd(U_node[i], left_edges=[U_node[i][0]], right_edges=[U_node[i][1]],max_singular_values=k[i])
    	U_,_,_=torch.svd(U[i])
    	U_node[i]=U_[:,:k[i]]
#TTM
#U_node[1].tensor = U_node[1].tensor.reshape(a,b,r)
U_node[1] = U_node[1].reshape(a,b,r)
B[1] = tn.ncon([U_node[1],U_node[3],U_node[4]],[(1,2,-3),(1,-1),(2,-2)])
B[0] = tn.ncon([U_node[0],U_node[1],U_node[2]],[(1,2,4,-3),(1,2,-1),(4,-2)])

#还原原始tensor
U1 = tn.ncon([B[1],U_node[3],U_node[4]],[(1,2,-3),(-1,1),(-2,2)])
U0_r = tn.ncon([B[0],U1,U_node[2]],[(1,2,-4),(-1,-2,1),(-3,2)])

#U0_r = torch.squeeze(U0_r.tensor)
U0_r = torch.squeeze(U0_r)
err=torch.norm(U[0] - U0_r)/torch.norm(U[0])
torch.cuda.synchronize()
end_time = time()
run_time = end_time-begin_time
print ('cost time is：',run_time)
print('err is ',err)
