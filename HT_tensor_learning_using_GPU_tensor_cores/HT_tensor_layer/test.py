from apex import amp
import torch.cuda.amp
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import StepLR

import torchvision
import torchvision.transforms as transforms
from torchvision import models

import tensorly as tl
import tensorly
from itertools import chain
from tensorly import unfold
from tensorly.decomposition import *
from scipy.linalg import svd
from scipy.linalg import norm
from torch.autograd import Variable
import os
import numpy as np
from time import *

transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=False, transform=transform_train)
# trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform_test)
# testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)

device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
cnttt = 0
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512,256)
        self.fc4 = nn.Linear(256,10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        #x = self.dropout(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        output = self.fc4(x)

        return output

class Net_ht(nn.Module):
    def __init__(self):
        super(Net_ht, self).__init__()
        r = 16
        #         self.fc1 = nn.Linear(784, 4096)
        self.u1 = nn.Parameter(torch.randn(4,16,r))
        self.b2 = nn.Parameter(torch.randn(r,r,r))
        self.u2 = nn.Parameter(torch.randn(7,16,r))
        self.b1 = nn.Parameter(torch.randn(r,r))
        self.u3 = nn.Parameter(torch.randn(28,16,r))        

        #self.fc2 = nn.Linear(4096, 4096)
        self.u1_2 = nn.Parameter(torch.randn(16,16,r))
        self.b2_2 = nn.Parameter(torch.randn(r,r,r))
        self.u2_2 = nn.Parameter(torch.randn(16,16,r))
        self.b1_2 = nn.Parameter(torch.randn(r,r))
        self.u3_2 = nn.Parameter(torch.randn(16,16,r))

        #self.fc3 = nn.Linear(4096, 4096)
        self.u1_3 = nn.Parameter(torch.randn(16,16,r))
        self.b2_3 = nn.Parameter(torch.randn(r,r,r))
        self.u2_3 = nn.Parameter(torch.randn(16,16,r))
        self.b1_3 = nn.Parameter(torch.randn(r,r))
        self.u3_3 = nn.Parameter(torch.randn(16,16,r))

        #self.fc4 = nn.Linear(4096, 4096)
        self.u1_4 = nn.Parameter(torch.randn(16,16,r))
        self.b2_4 = nn.Parameter(torch.randn(r,r,r))
        self.u2_4 = nn.Parameter(torch.randn(16,16,r))
        self.b1_4 = nn.Parameter(torch.randn(r,r))
        self.u3_4 = nn.Parameter(torch.randn(16,16,r))
        
        #self.fc5 = nn.Linear(4096, 256)
        self.u1_5 = nn.Parameter(torch.randn(16,4,r))
        self.b2_5 = nn.Parameter(torch.randn(r,r,r))
        self.u2_5 = nn.Parameter(torch.randn(16,8,r))
        self.b1_5 = nn.Parameter(torch.randn(r,r))
        self.u3_5 = nn.Parameter(torch.randn(16,8,r))
        
        self.fc6 = nn.Linear(256,10)
        
        self.dropout = nn.Dropout(p=0.5)
            

    def forward(self, x):
        batch = x.shape[0]
        x = torch.reshape(x,(batch,4,7,28))
        x = torch.einsum('zabc,ade->zbcde',(x,self.u1))
        x = torch.einsum('zabcd,def->zabcef',(x,self.b2))
        x = torch.einsum('zabcde,afd->zbcef',(x,self.u2))
        x = torch.einsum('zabcd,ce->zabde',(x,self.b1))
        x = torch.einsum('zabcd,aed->zbce',(x,self.u3))
        #x = torch.reshape(x,(batch,1024))
        #x = self.fc1(x)
        self.dropout = nn.Dropout(p=0.5)
        x = F.relu(x)

        x = torch.einsum('zabc,ade->zbcde',(x,self.u1_2))
        x = torch.einsum('zabcd,def->zabcef',(x,self.b2_2))
        x = torch.einsum('zabcde,afd->zbcef',(x,self.u2_2))
        x = torch.einsum('zabcd,ce->zabde',(x,self.b1_2))
        x = torch.einsum('zabcd,aed->zbce',(x,self.u3_2))
        self.dropout = nn.Dropout(p=0.5)
        #x = self.fc2(x)
        x = F.relu(x)

        x = torch.einsum('zabc,ade->zbcde',(x,self.u1_3))
        x = torch.einsum('zabcd,def->zabcef',(x,self.b2_3))
        x = torch.einsum('zabcde,afd->zbcef',(x,self.u2_3))
        x = torch.einsum('zabcd,ce->zabde',(x,self.b1_3))
        x = torch.einsum('zabcd,aed->zbce',(x,self.u3_3))
        self.dropout = nn.Dropout(p=0.5)
        #x = self.fc3(x)
        x = F.relu(x)

        x = torch.einsum('zabc,ade->zbcde',(x,self.u1_4))
        x = torch.einsum('zabcd,def->zabcef',(x,self.b2_4))
        x = torch.einsum('zabcde,afd->zbcef',(x,self.u2_4))
        x = torch.einsum('zabcd,ce->zabde',(x,self.b1_4))
        x = torch.einsum('zabcd,aed->zbce',(x,self.u3_4))
        self.dropout = nn.Dropout(p=0.5)
        #x = self.fc3(x)
        x = F.relu(x)

        x = torch.einsum('zabc,ade->zbcde',(x,self.u1_5))
        x = torch.einsum('zabcd,def->zabcef',(x,self.b2_5))
        x = torch.einsum('zabcde,afd->zbcef',(x,self.u2_5))
        x = torch.einsum('zabcd,ce->zabde',(x,self.b1_5))
        x = torch.einsum('zabcd,aed->zbce',(x,self.u3_5))
        self.dropout = nn.Dropout(p=0.5)
        #x = self.fc3(x)
        x = F.relu(x)


        x = torch.reshape(x,(batch,256))
        output = self.fc6(x)        
        return output


def ht(X,rank):
    #X=X.numpy()
    U=[0 for x in range(0,2)]
    B=[0 for x in range(0,1)]
    x_mat = unfold(X,0)
    U_,_,_=svd(x_mat)
    U[0]=U_[:,:rank[0]]
    
    
    x_mat = unfold(X,1)
    U_,_,_=svd(x_mat)
    U[1]=U_[:,:rank[1]]
    U[0]=torch.from_numpy(U[0])
    U[1]=torch.from_numpy(U[1])
    
    B[0] = tl.tenalg.multi_mode_dot(X,(U[0],U[1]),[0,1],transpose=True)
    
    #B[0]=torch.from_numpy(B[0])
    return U[0],U[1],B[0]

def ht_decomposition_fc_layer(layer, rank):
    l,r,core = ht(layer.weight.data, rank=rank)
    print(core.shape,l.shape,r.shape)
            
    right_layer = torch.nn.Linear(r.shape[0], r.shape[1])
    core_layer = torch.nn.Linear(core.shape[1], core.shape[0])
    left_layer = torch.nn.Linear(l.shape[1], l.shape[0])
    
    left_layer.bias.data = layer.bias.data
    left_layer.weight.data = l
    right_layer.weight.data = r.T

    new_layers = [right_layer, core_layer, left_layer]
    return nn.Sequential(*new_layers)

def decompose_fc():    
    model = torch.load("model")
    model.eval()
    model.cpu()
    for i, key in enumerate(model._modules.keys()):
        linear_layer = model._modules[key]
        if isinstance(linear_layer, torch.nn.modules.linear.Linear):
            rank = min(linear_layer.weight.data.numpy().shape) //8
            if (rank > 8):
            	model._modules[key] = ht_decomposition_fc_layer(linear_layer, [rank,rank])
        torch.save(model, 'model')
    return model

def build(decomp=True):
    print('==> Building model..')
    tl.set_backend('pytorch')
    full_net = Net()
    full_net = full_net.to(device)
    torch.save(full_net, 'model')
    if decomp:
        #decompose()
        decompose_fc()
    net = torch.load("model")
    print('==> Done')
    return net

def gen_noises(model,  layer_ids, std=1, co_matrices=None):
    noises = []
    for i, param in enumerate(model.parameters()):
        if i in layer_ids:
            if co_matrices == None:
                noises.append(torch.randn_like(param) * std) #生成与 param shape一样的随机tensor
            else:
                sz = co_matrices[i].shape[0]
                m = MultivariateNormal(torch.zeros(sz), co_matrices[i])
                noise = m.sample()
                noises.append(noise.reshape(param.shape))
        else:
            noises.append(torch.zeros_like(param))
        noises[-1] = noises[-1].to(device)
    return noises

def es_update(model, epsilons, ls, lr, layer_ids, mode=1, update=True):
    #         模型， 随机出来的tensor，根据随机tensor计算的 loss，学习率,层数【0~9】， mode=2 , updata = True
    device = epsilons[0][0].device
    num_directions = len(epsilons) #40
    elite_rate = 0.2
    elite_num = max(int(elite_rate * num_directions), 1)  #8

    ls = torch.tensor(ls).to(device)
    if mode == 1:
        weight = ls
    else:
        weight = 1 / (ls + 1e-8)
    indices = torch.argsort(weight)[-elite_num:]
    mask = torch.zeros_like(weight)
    mask[indices] = 1

    weight *= mask
    weight = weight / torch.sum(weight)

    grad = []
    for l in epsilons[0]:
        grad.append(torch.zeros_like(l))

    for idx in indices:
        for i, g in enumerate(epsilons[idx]):
            grad[i] += g * weight[idx]
    if update:
        if mode==1:
            i = 0
            for g, param in zip(grad, model.parameters()):
                if i in layer_ids:
                    param.requires_grad = False
                    param -= lr * g
                    param.requires_grad = True
                i += 1
        else:
            i = 0
            # print(len(grad), layer_ids)
            for g, param in zip(grad, model.parameters()):
                if i in layer_ids:
                    # print("update")
                    param.requires_grad = False
                    param += lr * g
                    param.requires_grad = True
                i += 1

    return grad



def add_noises(model, noises, layer_ids):
    i = 0
    for param, noise in zip(model.parameters(), noises):
        if i in layer_ids:
            param.requires_grad = False
            param += noise
            param.requires_grad = True
        i += 1



def remove_noises(model, noises, layer_ids):
    i = 0
    for param, noise in zip(model.parameters(), noises):
        if i in layer_ids:
            param.requires_grad = False
            param -= noise
            param.requires_grad = True
        i += 1

def explore_one_direction(model, inputs, targets, criterion, layer_ids, co_matrices, return_list, if_mirror):
    #               模型   输入[400,1,28,28] 标签：400 ，损失函数，[0~9]   None     用于写入文件  result  True
    ep_rt = []
    ls_rt = []

    epsilon = gen_noises(model, layer_ids, std=0.01, co_matrices=co_matrices) #epsilon len(layer_ids) 随机的tensor
    add_noises(model, epsilon, layer_ids) # 权重矩阵 = 权重矩阵 + 随机矩阵

    with torch.cuda.amp.autocast():   	
        outputs = model(inputs) # 前向

    loss = criterion(outputs, targets).item()  #计算损失 （item-> 将tensor 转化为浮点数）
    remove_noises(model, epsilon, layer_ids)  # 权重矩阵 = 权重矩阵 - 随机矩阵

    ep_rt.append(epsilon.copy()) #随机矩阵 list
    ls_rt.append(loss)  # loss 的list

    if if_mirror:        
        for i in range(len(epsilon)): # 每个随机 tensor取数相反数
            epsilon[i] = -epsilon[i]
        add_noises(model, epsilon, layer_ids) #添加 随机数

        with torch.cuda.amp.autocast(): 
            outputs = model(inputs)  #前向

        loss = criterion(outputs, targets).item() #计算loss
        remove_noises(model, epsilon, layer_ids) #移除随机数
        ep_rt.append(epsilon)
        ls_rt.append(loss)

    return ep_rt, ls_rt

def test(test_acc, best_acc, model):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        print('|', end='')
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if batch_idx % 10 == 0:
                print('=', end='')
    acc = 100. * correct / total
    print('|', 'Accuracy:', acc, '% ', correct, '/', total)
    test_acc.append(correct / total)
    return max(acc, best_acc)


#ht_net = build()
ht_net = Net_ht()
model = ht_net
num_epoch = 10
lr = 0.5
lr0 = lr
step_size = 3
gamma = 0.5
co_matrices = None
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
optimizer = optim.SGD(model.parameters(), lr=0.001) #优化函数
    # scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
criterion = nn.CrossEntropyLoss() #损失函数
    # print(model.parameters())
num_layers = len(model.state_dict()) # 有 num_layers 层  10

train_acc = []
test_acc = []
best_acc = 0
global_mean = []
global_var = []

model.train()
model = model.to(device)
model.share_memory()
early_break = False

es_mode = 2
num_directions = 40
num_directions0 = num_directions

if_alternate = False
fall_cnt = 0
if_es = True
if_bp = False
if_mirror = True

layer_ids = list(range(num_layers))   # 0~9  一共是10层
num_directions = num_directions0   #40
if if_mirror:
    num_directions = num_directions // 2 #  变成  20
lr = lr0  #  0.5

#ht_net, optimizer = amp.initialize(ht_net, optimizer, opt_level="O1")

# torch.cuda.synchronize()
# begin_time = time()

for epoch in range(num_epoch):
    print("\nES layer ", "alternate" if if_alternate else layer_ids, "  Epoch: {}".format(epoch))
    print("|", end="")
    train_loss = 0
    correct = 0
    total = 0            
    epoch_mean = []
    epoch_var = []

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if(batch_idx == int(60000/64)):
            break
        if if_alternate:
            layer_ids = [layer_id]
            layer_id = (layer_id + 1) % num_layers
            print("if_alternate if perform!~~~")
        total += len(inputs)
        ls = []
        epsilons = []
        processes = []
        result = []

        inputs, targets = inputs.to(device), targets.to(device)
        inputs = Variable(inputs)
        inputs.requires_grad = True

        for _ in range(num_directions):
            epsilon, loss = explore_one_direction(model, inputs, targets, criterion, layer_ids, co_matrices, result, if_mirror)
            epsilons.extend(epsilon)  # 每次循环里面有两个 epsilon，一正，一个相反数
            ls.extend(loss)   # 每次也就有两个 loss
            for l in loss:
                train_loss += l

        es_grad = es_update(model, epsilons, ls, lr, layer_ids, es_mode, update=if_es)

        with torch.cuda.amp.autocast(): 
            outputs = model(inputs)
            
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        if batch_idx % 10 == 0:
            print('=', end='')
    print('|', 'Accuracy:', 100. * correct / total, '% ', correct, '/', total)
    best_acc = test(test_acc, best_acc, model)

    if epoch % step_size == 0 and epoch:
        lr *= gamma
        lr = max(lr, 0.0125)
        if epoch % (step_size * 2) == 0: 
            num_directions = max(int(num_directions/gamma), num_directions + 1)
        pass
    train_acc.append(correct / total)

    if epoch >= 2:
        if train_acc[-1] - train_acc[-2] < 0.01 and train_acc[-2] - train_acc[-3] < 0.01:
            fall_cnt += 1
        else:
            fall_cnt = 0

    print('Current learning rate: ', lr)
    print('Current num_directions: ', num_directions, "mirror" if if_mirror else "")

    print('Best training accuracy overall: ', best_acc)

# torch.cuda.synchronize()
# end_time = time()
# run_time = end_time-begin_time
# print ('cost time is：',run_time)