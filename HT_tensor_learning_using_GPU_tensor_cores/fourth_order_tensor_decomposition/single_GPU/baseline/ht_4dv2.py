import tensorly as tl
from tensorly import random
from tensorly import unfold
from tensorly import fold
from tensorly import *
import torch
from time import *
import numpy as np


tl.set_backend('pytorch')

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

U[2] = tl.tenalg.multi_mode_dot(B[2],(U[5],U[6]),[0,1],transpose=False)
U[1] = tl.tenalg.multi_mode_dot(B[1],(U[3],U[4]),[0,1],transpose=False)
leaf=np.size(U[3],0)
right = np.size(U[4],0)
U[1] = fold(U[1].T,2,(leaf,right,k[1]))

leaf=np.size(U[5],0)
right = np.size(U[6],0)
U[2] = fold(U[2].T,2,(leaf,right,k[2]))

U[0] = tl.tenalg.multi_mode_dot(B[0],(U[1],U[2]),[0,1],transpose=False)
U[0] = torch.squeeze(U[0])


#torch.cuda.synchronize()
begin_time = time()
print("size:\n",n)
U[3] = U[0].permute(0,1,2,3).reshape(a,b*c*d)
U[4] = U[0].permute(1,0,2,3).reshape(b,a*c*d)
U[5] = U[0].permute(2,0,1,3).reshape(c,a*b*d)
U[6] = U[0].permute(3,0,1,2).reshape(d,a*b*c)

U[1] = U[0].permute(0,1,2,3).reshape(a*b,c*d)
U[2] = U[0].permute(2,3,0,1).reshape(c*d,a*b)

for i in range(6,-1,-1):
    if i ==0: #root
        U[i]=tensor_to_vec(X)
    elif i==1:
        x_mat = U[0].reshape(a*b,c*d)
        U[1],_,_ = tl.truncated_svd(x_mat,k[1])
    elif i ==2:
        x_mat = U[0].reshape(a*b,c*d).T
        U[2],_,_ = tl.truncated_svd(x_mat,k[2])

        
    else:     #叶子结点
        x_mat=U[i]
        x_mat = torch.matmul(x_mat,x_mat.T)
        _,U_=torch.eig(x_mat)
        #U_ = U_.float()
        #U_,_,_=svd(x_mat)
        U[i]=U_[:,:k[i]]

    if i < 2:
        leaf=np.size(U[2*i+1],0)
        right = np.size(U[2*i+2],0)
        U_tensor = fold(U[i].T,2,(leaf,right,k[i]))
        B[i]=tl.tenalg.multi_mode_dot(U_tensor,(U[2*i+1],U[2*i+2]),[0,1],transpose=True)


for i in range(2,-1,-1):
    U[i] = tl.tenalg.multi_mode_dot(B[i],(U[2*i+1],U[2*i+2]),[0,1],transpose=False) #B1 U3 U4
    U[i]=unfold(U[i],2)
    U[i]=U[i].T
U[0] = U[0].reshape(a,b,c,d)

r=vec_to_tensor(U[0],[n,n,n,n])
err = torch.norm(X-r)/torch.norm(X)
#torch.cuda.synchronize()
end_time = time()
run_time = end_time-begin_time
print ('cost time is：',run_time)
print('err is ',err)

