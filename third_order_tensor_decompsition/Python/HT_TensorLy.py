import numpy as np
from scipy.linalg import svd
from scipy.linalg import norm
import scipy.io as sio
from collections import deque
import tensorly as tl
import numpy as np
from tensorly.decomposition import *
from tensorly import *
#import jax.numpy as jnp
import torch
from time import *


svd = torch.svd
eig = torch.eig


n=160
r = int(n*0.1)
U=[0 for x in range(0,5)]
B=[0 for x in range(0,5)]
dim=[[0,1,2],[0,1],2,0,1]
k=[1,r,r,r,r]

tl.set_backend('pytorch')
for i in range(4,1,-1):
    U[i] = np.random.rand(n,k[i]).astype(np.float32)
    U[i] = tl.tensor(U[i],device='cuda:0')

B[1] = np.random.rand(k[3],k[4],k[1]).astype(np.float32)
B[0] = np.random.rand(k[1],k[2],k[0]).astype(np.float32)

B[1] = tl.tensor(B[1],device='cuda:0')
B[0] = tl.tensor(B[0],device='cuda:0')
  
for i in range(1,-1,-1):
    U[i] = tl.tenalg.multi_mode_dot(B[i],(U[2*i+1],U[2*i+2]),[0,1],transpose=False) #B1 U3 U4
    U[i]=unfold(U[i],2)
    U[i]=U[i].T

X=U[0].reshape(n,n,n)


#factors = parafac(X, rank=r,n_iter_max=10,init = 'random',tol=10e-6)
#X = tl.kruskal_to_tensor(factors)


print("size:\n",n)
torch.cuda.synchronize()
begin_time = time()
for i in range(4,-1,-1):
    if i ==0: #root
        U[i]=tensor_to_vec(X)
    elif i==1:
        x_mat = unfold(X,2)
        x_mat=x_mat.T
        U_,_,_=svd(x_mat)
        U[i]=U_[:,:k[i]]

    else:     #叶子结点
        x_mat=unfold(X,dim[i])
        x_mat = torch.matmul(x_mat,x_mat.T)
        _,U_=eig(x_mat,True)
        #U_,_,_=svd(x_mat)
        U[i]=U_[:,:k[i]]

    if i < 2:
        leaf=np.size(U[2*i+1],0)
        right = np.size(U[2*i+2],0)
        U_tensor = fold(U[i].T,2,(leaf,right,k[i]))
        B[i]=tl.tenalg.multi_mode_dot(U_tensor,(U[2*i+1],U[2*i+2]),[0,1],transpose=True)


#还原
for i in range(1,-1,-1):
	U[i] = tl.tenalg.multi_mode_dot(B[i],(U[2*i+1],U[2*i+2]),[0,1],transpose=False) #B1 U3 U4
	U[i]=unfold(U[i],2)
	U[i]=U[i].T

r=vec_to_tensor(U[0],[n,n,n])
err = torch.norm(X-r)/torch.norm(X)
torch.cuda.synchronize()
end_time = time()
run_time = end_time-begin_time
print ('cost time is：',run_time)
print('err is ',err)






