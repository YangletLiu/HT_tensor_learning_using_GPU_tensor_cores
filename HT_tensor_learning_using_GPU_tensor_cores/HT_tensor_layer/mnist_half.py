######################### 0. import packages #############################
from apex import amp
import torch.cuda.amp
import torch
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms

import  matplotlib.pyplot as plt
import tensorly as tl
import tensorly
from itertools import chain
from tensorly.decomposition import parafac, partial_tucker, tucker

import numpy as np
import time
import sys
import os

import tensornetwork as tn
tn.set_default_backend("pytorch")
svd = torch.svd
eig = torch.eig

########################## 1. load data ####################################
device = 'cuda:4' if torch.cuda.is_available() else 'cpu'

# transform_train = torchvision.transforms.Compose([
#                                 torchvision.transforms.Pad(padding=2),
#                                 torchvision.transforms.ToTensor(),
#                                 torchvision.transforms.Normalize(
#                                     (0.1307,), (0.3081,))
#                             ])

# transform_test = torchvision.transforms.Compose([
#                                 torchvision.transforms.Pad(padding=2),
#                                 torchvision.transforms.ToTensor(),
#                                 torchvision.transforms.Normalize(
#                                     (0.1307,), (0.3081,))
#                              ])

# batch_size = 128
# trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
# num_train = len(trainset)

# testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
# testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)
# num_test = len(testset)

transform_train = torchvision.transforms.Compose([
                                torchvision.transforms.Pad(padding=2),
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                             ])

transform_test = torchvision.transforms.Compose([
                                torchvision.transforms.Pad(padding=2),
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                             ])

batch_size = 128
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
train_X = torch.from_numpy(np.vstack([_[0].reshape(-1, 32, 32) for _  in trainloader])).to(device)
train_X = torch.unsqueeze(train_X, dim=1)
# print(train_X.shape)
# yy = [_[1].reshape(-1) for _  in trainloader]
# print(yy[0].shape)
train_Y = torch.from_numpy(np.hstack([_[1].reshape(-1) for _ in trainloader])).to(device)
# print(train_Y.shape)
# exit(0)
num_train = len(trainset)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
# testloader = [(_[0].to(device), _[1].to(device)) for _ in testloader]
num_test = len(testset)
########################### 2. define model ##################################
def get_num_params(net, name=""):
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("-{} #param: {}".format(name, trainable_num))
    return


# num_param: 83658 -> 40096
class FCNet(nn.Module):
    def __init__(self):
        super(FCNet, self).__init__()
        self.fc1 = nn.Linear(32*32, 1024)
        self.fc2 = nn.Linear(1024, 10)
        #self.fc3 = nn.Linear(4096, 4096)
        #self.fc4 = nn.Linear(4096, 4096)
        # self.fc5 = nn.Linear(1024, 512)
        # self.fc6 = nn.Linear(512,10)


    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        x = F.relu(x)
        
        x = self.fc2(x)

        # x = F.relu(x)
        
        # x = self.fc5(x)
        # x = F.relu(x)
        
        # x = self.fc6(x)
        return x

class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class Net_ht(nn.Module):
    def __init__(self):
        super(Net_ht, self).__init__()
        r = 16
        #self.fc1 = nn.Linear(784, 1024)
        self.u1_1 = nn.Parameter(torch.randn(8,8,r))
        self.b2_1 = nn.Parameter(torch.randn(r,r,r))
        self.u2_1 = nn.Parameter(torch.randn(8,8,r))
        self.b1_1 = nn.Parameter(torch.randn(r,r))
        self.u3_1 = nn.Parameter(torch.randn(16,16,r))
        
        self.b1 = nn.Parameter(torch.randn(8,8,16))

        #self.fc2 = nn.Linear(1024, 1024)
        self.u1_2 = nn.Parameter(torch.randn(8,8,r))
        self.b2_2 = nn.Parameter(torch.randn(r,r,r))
        self.u2_2 = nn.Parameter(torch.randn(8,8,r))
        self.b1_2 = nn.Parameter(torch.randn(r,r))
        self.u3_2 = nn.Parameter(torch.randn(16,16,r))
        
        self.b2 = nn.Parameter(torch.randn(8,8,16))

        #self.fc3 = nn.Linear(1024, 512)
        self.u1_3 = nn.Parameter(torch.randn(8,8,r))
        self.b2_3 = nn.Parameter(torch.randn(r,r,r))
        self.u2_3 = nn.Parameter(torch.randn(8,8,r))
        self.b1_3 = nn.Parameter(torch.randn(r,r))
        self.u3_3 = nn.Parameter(torch.randn(16,8,r))
        
        self.b3 = nn.Parameter(torch.randn(8,8,8))
       
        self.fc6 = nn.Linear(512,10)
        
        self.dropout = nn.Dropout(p=0.5)
            

    def forward(self, x):
        batch = x.shape[0]
        x = torch.reshape(x,(batch,8,8,16))
        x = torch.einsum('zabc,ade->zbcde',(x,self.u1_1))
        x = torch.einsum('zabcd,def->zabcef',(x,self.b2_1))
        x = torch.einsum('zabcde,afd->zbcef',(x,self.u2_1))
        x = torch.einsum('zabcd,ce->zabde',(x,self.b1_1))
        x = torch.einsum('zabcd,aed->zbce',(x,self.u3_1))
        x = x + self.b1
        #x = torch.reshape(x,(batch,1024))
        #x = self.fc1(x)
        self.dropout = nn.Dropout(p=0.5)
        x = F.relu(x)

        x = torch.einsum('zabc,ade->zbcde',(x,self.u1_2))
        x = torch.einsum('zabcd,def->zabcef',(x,self.b2_2))
        x = torch.einsum('zabcde,afd->zbcef',(x,self.u2_2))
        x = torch.einsum('zabcd,ce->zabde',(x,self.b1_2))
        x = torch.einsum('zabcd,aed->zbce',(x,self.u3_2))
        x = x + self.b2
        self.dropout = nn.Dropout(p=0.5)
        #x = self.fc2(x)
        x = F.relu(x)

        x = torch.einsum('zabc,ade->zbcde',(x,self.u1_3))
        x = torch.einsum('zabcd,def->zabcef',(x,self.b2_3))
        x = torch.einsum('zabcde,afd->zbcef',(x,self.u2_3))
        x = torch.einsum('zabcd,ce->zabde',(x,self.b1_3))
        x = torch.einsum('zabcd,aed->zbce',(x,self.u3_3))
        x = x + self.b3
        self.dropout = nn.Dropout(p=0.5)
        #x = self.fc3(x)
        x = F.relu(x)


        x = torch.reshape(x,(batch,512))
        output = self.fc6(x)        
        return output
def ht(data,rank,N,M):
    a = N[0]*M[0]
    b = N[1]*M[1]
    c = N[2]*M[2]
    X = torch.reshape(data,(a,b,c))
    
    U=[0 for x in range(0,5)]
    U[0] = X
    U[3] = U[0].permute(0,1,2).reshape(a,b*c)
    U[4] = U[0].permute(1,0,2).reshape(b,a*c)
    U[2] = U[0].permute(2,0,1).reshape(c,a*b)
    U[1] = U[0].permute(0,1,2).reshape(a*b,c)
    
    U_node=[0 for x in range(0,5)]
    B=[0 for x in range(0,5)]
    k=[1,rank,rank,rank,rank]
    dim=[[0,1,2],[0,1],2,0,1]
    for i in range(4,0,-1):
        if i>1:
            x_mat = torch.matmul(U[i],U[i].T)
            _,U_=torch.eig(x_mat,True)
            U_node[i]=U_[:,:k[i]]
        else:
            U_,_,_=torch.svd(U[i])
            U_node[i]=U_[:,:k[i]]
    U_node[1] = U_node[1].reshape(a,b,rank)
    U_node[0] = U[0].reshape(a,b,c,1)    
    B[1] = tn.ncon([U_node[1],U_node[3],U_node[4]],[(1,2,-3),(1,-1),(2,-2)])  
    B[0] = tn.ncon([U_node[0],U_node[1],U_node[2]],[(1,2,4,-3),(1,2,-1),(4,-2)])
    return U_node[3].reshape(N[0],M[0],rank),U_node[4].reshape(N[1],M[1],rank),U_node[2].reshape(N[2],M[2],rank),B[1],B[0].reshape(rank,rank)    
class tNet(nn.Module):
    def __init__(self):
        super(tNet, self).__init__()
        self.fc1 = nn.Linear(32*32, 1024)
        [u1,u2,u3,b2,b1] = ht(self.fc1.weight.data,16,[8,8,16],[8,8,16])
        self.u1_1 = nn.Parameter(u1)
        self.u2_1 = nn.Parameter(u2)
        self.u3_1 = nn.Parameter(u3)
        self.b2_1 = nn.Parameter(b2)
        self.b1_1 = nn.Parameter(b1)
        self.b1 = nn.Parameter(self.fc1.bias.data.reshape(8,8,16))
        
        self.fc2 = nn.Linear(1024, 10)
        # [u1,u2,u3,b2,b1] = ht(self.fc2.weight.data,16,[8,8,16],[8,8,16])
        # self.u1_2 = nn.Parameter(u1)
        # self.u2_2 = nn.Parameter(u2)
        # self.u3_2 = nn.Parameter(u3)
        # self.b2_2 = nn.Parameter(b2)
        # self.b1_2 = nn.Parameter(b1)
        # self.b2 = nn.Parameter(self.fc2.bias.data.reshape(8,8,16))
        
        # self.fc3 = nn.Linear(1024, 512)
        # [u1,u2,u3,b2,b1] = ht(self.fc3.weight.data,16,[8,8,16],[8,8,8])
        # self.u1_3 = nn.Parameter(u1)
        # self.u2_3 = nn.Parameter(u2)
        # self.u3_3 = nn.Parameter(u3)
        # self.b2_3 = nn.Parameter(b2)
        # self.b1_3 = nn.Parameter(b1)
        # self.b3 = nn.Parameter(self.fc3.bias.data.reshape(8,8,8))
        
        # self.fc4 = nn.Linear(512,10)
        
        
    def forward(self, x):
        batch = x.shape[0]
        x = torch.reshape(x,(batch,8,8,16))
        x = torch.einsum('zabc,ade->zbcde',(x,self.u1_1))
        x = torch.einsum('zabcd,def->zabcef',(x,self.b2_1))
        x = torch.einsum('zabcde,afd->zbcef',(x,self.u2_1))
        x = torch.einsum('zabcd,ce->zabde',(x,self.b1_1))
        x = torch.einsum('zabcd,aed->zbce',(x,self.u3_1))
        x = x + self.b1
        x = torch.reshape(x,(batch,1024))
        #x = self.fc1(x)
        
        x = F.relu(x)
        
        # x = torch.einsum('zabc,ade->zbcde',(x,self.u1_2))
        # x = torch.einsum('zabcd,def->zabcef',(x,self.b2_2))
        # x = torch.einsum('zabcde,afd->zbcef',(x,self.u2_2))
        # x = torch.einsum('zabcd,ce->zabde',(x,self.b1_2))
        # x = torch.einsum('zabcd,aed->zbce',(x,self.u3_2))
        # x = x + self.b2

        # #x = self.fc2(x)
        # x = F.relu(x)
        
        #x = torch.reshape(x,(batch,1024))
        # x = torch.einsum('zabc,ade->zbcde',(x,self.u1_3))
        # x = torch.einsum('zabcd,def->zabcef',(x,self.b2_3))
        # x = torch.einsum('zabcde,afd->zbcef',(x,self.u2_3))
        # x = torch.einsum('zabcd,ce->zabde',(x,self.b1_3))
        # x = torch.einsum('zabcd,aed->zbce',(x,self.u3_3))
        # x = x + self.b3
        output = self.fc2(x)
        #x = F.relu(x)
        
        # x = torch.reshape(x,(batch,512))
        #output = self.fc4(x)
        
        return output

######################## 3. build model functions #################
def weight_init(m):
    if isinstance(m,nn.Linear):
        # nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
        nn.init.xavier_normal_(m.weight)


def W_decomposition_fc_layer(layer, rank):
    layer_shape = layer.weight.data.shape
    # y = Wx  ==>  y = W1W2x
    W1 = nn.Linear(in_features=layer_shape[1],
                   out_features=rank,
                   bias=False)
    W2 = nn.Linear(in_features=rank,
                   out_features=layer_shape[0],
                   bias=False)
    new_layers = [W1,  W2]
    return nn.Sequential(*new_layers)


def W_decompose_nested_layer(layer):
    modules = layer._modules
    for key in modules.keys():
        l = modules[key]
        if isinstance(l, nn.Sequential):
            modules[key] = W_decompose_nested_layer(l)
        elif isinstance(l, nn.Linear):
            print("ok")
            fc_layer = l
            sp = fc_layer.weight.data.numpy().shape
            rank = min(max(sp)//8, min(sp))
            modules[key] = W_decomposition_fc_layer(fc_layer, rank)
    return layer


# decomposition
def W_decompose(model):
    model.eval()
    model.cpu()
    layers = model._modules # model.features._modules  # model._modules
    for i, key in enumerate(layers.keys()):
        # if i >= len(layers.keys()):
        #     break
        if isinstance(layers[key], torch.nn.modules.Linear):
            print("ok")
            fc_layer = layers[key]
            # rank = max(fc_layer.weight.data.numpy().shape) // 10
            sp = fc_layer.weight.data.numpy().shape
            rank = min(max(sp)//8, min(sp))
            layers[key] = W_decomposition_fc_layer(fc_layer, rank)
        elif isinstance(layers[key], nn.Sequential):
            layers[key] = W_decompose_nested_layer(layers[key])
    return model


# build model
def build(decomp=True):
    print('==> Building model..')
    tl.set_backend('pytorch')
    #full_net = FCNet()
    full_net = tNet()
    # full_net.fc1 = Identity()
    # full_net.fc2 = Identity()
    # full_net.fc3 = Identity()
    get_num_params(full_net, "nd FC net")
    # print(full_net)
    full_net = full_net.to(device)
    if decomp:
        full_net = W_decompose(full_net)
    else:
        print("NO decomp")
    full_net.apply(weight_init)
    print('==> Done')
    return full_net

########################### 4. train and test functions #########################
criterion = nn.CrossEntropyLoss()
lr0 = 0.0005
std = 0.01

def query_lr(epoch):
    lr = lr0
    # if epoch >= 20:
    #     lr *= 0.2 ** 3
    # elif epoch >= 15:
    #     lr *= 0.2 ** 2
    # elif epoch >= 8:
    #     lr *= 0.2 ** 1
    # else:
    #     lr *= 0.2 ** 0
    return lr


def set_lr(optimizer, epoch):
    current_lr = query_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr
    return current_lr


def test(epoch, net, best_acc, test_acc_list, test_loss_list):
    net.eval()
    net.training = False
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = Variable(inputs), Variable(targets)
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

        # Save checkpoint when best model
        acc = 100.*correct/total
        print("\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc: %.2f%%" %(epoch, loss.item(), acc))

        if acc > best_acc:
            print('| Saving Best model...\t\t\tTop1 = %.2f%%' %(acc))
            best_acc = acc
        test_acc_list.append(acc)
        test_loss_list.append(test_loss / num_test)
    return best_acc


# Training
def bp_train(num_epochs, net):
    net = net.to(device)
    train_acc_list, train_loss_list = [], []
    test_acc_list, test_loss_list = [], []
    best_acc = 0.

    original_time = time.asctime(time.localtime(time.time()))
    start_time = time.time()

    optimizer = SGD(net.parameters(), lr=lr0, momentum=0.9, weight_decay=5e-4)
    # optimizer = Adam(net.parameters(), lr=lr0)
    # scheduler = StepLR(optimizer, step_size=20, gamma=0.9)
    current_lr = lr0


    num_batch = num_train//batch_size + bool(num_train%batch_size)
    shuffle_train = np.array([np.random.permutation(len(train_X)) for i in range(num_epochs)])
    shuffle_train = torch.from_numpy(shuffle_train.astype(np.int64)).to(device)


    try:
        for epoch in range(num_epochs):
            net.train()
            net.training = True
            train_loss = 0
            correct = 0
            total = 0

            current_lr = set_lr(optimizer, epoch)
            print('\n=> Training Epoch #%d, LR=%.4f' %(epoch, current_lr))
            # for batch_idx, (inputs, targets) in enumerate(trainloader):
            for batch_idx in range(num_batch):
                # if proc_id == 0:
                #     start_ = time.time()
                inputs = train_X[shuffle_train[epoch, batch_idx*batch_size:batch_idx*batch_size+batch_size]]
                targets = train_Y[shuffle_train[epoch, batch_idx*batch_size:batch_idx*batch_size+batch_size]]
                inputs, targets = Variable(inputs), Variable(targets)
                # if proc_id == 0:
                #     print(inputs.device, targets.device)
                inputs, targets = inputs.to(device), targets.to(device) # GPU settings
                inputs, targets = Variable(inputs), Variable(targets)
                inputs, targets = inputs.to(device), targets.to(device) # GPU settings
                optimizer.zero_grad()

                
                outputs = net(inputs)               # Forward Propagation

                loss = criterion(outputs, targets)  # Loss
                loss.backward()  # Backward Propagation
                optimizer.step() # Optimizer update

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += predicted.eq(targets.data).cpu().sum()

                sys.stdout.write('\r')
                sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc: %.3f%%'
                        %(epoch, num_epochs, batch_idx+1,
                            (len(trainset)//batch_size)+1, loss.item(), 100.*correct/total))
                sys.stdout.flush()

            # scheduler.step()
            # current_lr = scheduler.get_last_lr()[0]  # query_lr(epoch)
            best_acc = test(epoch, net, best_acc, test_acc_list, test_loss_list)
            train_acc_list.append(100.*correct/total)
            train_loss_list.append(train_loss / num_train)
            now_time = time.time()
            print("Used:{}s \t EST: {}s".format(now_time-start_time, (now_time-start_time)/(epoch+1)*(num_epochs-epoch-1)))
    except KeyboardInterrupt:
        pass

    end_time = time.time()
    print("\nBest training accuracy overall: ", best_acc)

    dir_path = "./mnist_backups_half/{}_{:.3f}".format(original_time, best_acc)
    os.mkdir(dir_path)
    save_record_and_draw(dir_path, (train_acc_list, train_loss_list), (test_acc_list, test_loss_list), [end_time-start_time, epoch])
    print("- Record and accuracy, loss figs are saved to {}".format(dir_path))
    return 


def get_indices(net):
    indices = [0]
    for param in net.parameters():
        indices.append(np.prod(list(param.data.shape))+indices[-1])
    indices = torch.tensor(indices, device=device)
    return indices


def get_flatten_parameters(net):
    params = []
    for param in net.parameters():
        params.append(*param.reshape(-1))
    return torch.tensor(params, device=device)


def set_flatten_parameters(net, params):
    for i, param in enumerate(net.parameters()):
        param = params[i:i+1].reshape(param.shape)
    return 


def gen_noise(net):
    noises = []
    for param in net.parameters():
        noises.append(torch.randn_like(param, device=device) * std)
    return noises


def add_noise(model, noises):
    for param, noise in zip(model.parameters(), noises):
        param.requires_grad = False
        param += noise
        param.requires_grad = True


def remove_noise(model, noises):
    for param, noise in zip(model.parameters(), noises):
        param.requires_grad = False
        param -= noise
        param.requires_grad = True


def explore_one_direction(net, data, if_mirror):
    inputs, targets = data
    noise = gen_noise(net)

    add_noise(net, noise)

    with torch.cuda.amp.autocast():
        outputs = net(inputs)

    remove_noise(net, noise)
    loss = criterion(outputs, targets).item()
    if if_mirror:
        inverse_noise = []
        for i in range(len(noise)):
            inverse_noise.append(-noise[i])
        add_noise(net, inverse_noise)

        with torch.cuda.amp.autocast():
            outputs = net(inputs)

        remove_noise(net, inverse_noise)
        loss -= criterion(outputs, targets).item()

    return loss, noise


def get_es_grad(loss_list, noise_list, num, mode="loss") -> list:
    loss_list = torch.tensor(loss_list, device=device)
    loss_list /= batch_size

    if mode == "loss":
        weight = loss_list
        coefficient = 1
    elif mode == "score":
        weight = 1. / (loss_list + 1e-8)
        coefficient = -1
    indices = torch.argsort(weight)[-num:]
    # weight /= weight[indices].sum()

    grad = []
    for i in range(len(noise_list[0])):
        grad.append(torch.zeros_like(noise_list[0][i], device=device))
    for idx in indices:
        for i, g in enumerate(noise_list[idx]):
            grad[i] += coefficient * g * weight[idx] / (num * std)

    return grad


def set_es_grad(net, grad):
    for param, g in zip(net.parameters(), grad):
        param.grad = g
    return


def es_train(num_epochs, net):
    net = net.to(device)
    # for param in net.parameters():
    #     param.requires_grad = False
    train_acc_list, train_loss_list = [], []
    test_acc_list, test_loss_list = [], []
    best_acc = 0.

    original_time = time.asctime(time.localtime(time.time()))
    start_time = time.time()

    # optimizer = SGD(net.parameters(), lr=lr0, momentum=0.9, weight_decay=5e-4)
    # optimizer = SGD(net.parameters(), lr=lr0)
    optimizer = Adam(net.parameters(), lr=lr0)
    # scheduler = StepLR(optimizer, step_size=20, gamma=0.9)
    current_lr = lr0

    num_directions = 1000
    elite_rate = 1.
    if_mirror = True
    elite_num = max(int(elite_rate * num_directions), 1)
    try:
        for epoch in range(num_epochs):
            net.train()
            net.training = True
            train_loss = 0
            correct = 0
            total = 0


            current_lr = set_lr(optimizer, epoch)
            print('\n=> Training Epoch #%d, LR=%.4f, std=%.5f, num_directions=%d %s, elite_rate=%.2f' %(epoch, current_lr, std, num_directions, "mirror" if if_mirror else "", elite_rate))
            for batch_idx, (inputs, targets) in enumerate(trainloader):

                inputs, targets = Variable(inputs), Variable(targets)
                inputs, targets = inputs.to(device), targets.to(device) # GPU settings

                # get loss
                with torch.cuda.amp.autocast(): 
                    outputs = net(inputs)               # Forward Propagation

                loss = criterion(outputs, targets)  # Loss
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += predicted.eq(targets.data).cpu().sum()

                # update
                optimizer.zero_grad()
                # similar to loss.backward
                loss_list = []
                noise_list = []
                for _ in range(num_directions):
                    l, noise = explore_one_direction(net, (inputs, targets), if_mirror=if_mirror)
                    loss_list.append(l)
                    noise_list.append(noise)
                grad = get_es_grad(loss_list, noise_list, elite_num)
                set_es_grad(net, grad)

                optimizer.step()

                sys.stdout.write('\r')
                sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc: %.3f%%'
                        %(epoch, num_epochs, batch_idx+1,
                            (len(trainset)//batch_size)+1, loss.item(), 100.*correct/total))
                sys.stdout.flush()

            # scheduler.step()
            # current_lr = scheduler.get_last_lr()[0]  # query_lr(epoch)
            best_acc = test(epoch, net, best_acc, test_acc_list, test_loss_list)
            train_acc_list.append(100.*correct/total)
            train_loss_list.append(train_loss / num_train)
            now_time = time.time()
            print("Used:{}s \t EST: {}s".format(now_time-start_time, (now_time-start_time)/(epoch+1)*(num_epochs-epoch-1)))
    except KeyboardInterrupt:
        pass
    end_time = time.time()
    print("\nBest training accuracy overall: ", best_acc)

    dir_path = "./mnist_backups_half/{}_{:.3f}".format(original_time, best_acc)
    os.mkdir(dir_path)
    save_record_and_draw(dir_path, (train_acc_list, train_loss_list), (test_acc_list, test_loss_list), [end_time-start_time, epoch])
    print("- Record and accuracy, loss figs are saved to {}".format(dir_path))
    return 


def save_record_and_draw(dir_path, train_record, test_record, others):
    train_acc_list, train_loss_list = train_record
    test_acc_list, test_loss_list = test_record
    cost_time, epoch = others

    np.savetxt(dir_path+"/log.txt", others)
    np.savetxt(dir_path+"/train_loss_list.txt", train_loss_list)
    np.savetxt(dir_path+"/test_loss_list.txt", test_loss_list)
    np.savetxt(dir_path+"/train_acc_list.txt", train_acc_list)
    np.savetxt(dir_path+"/test_acc_list.txt", test_acc_list)

    plt.cla()
    plt.plot(range(len(train_loss_list)), train_loss_list, label="train")
    plt.plot(range(len(test_loss_list)), test_loss_list, label="test")
    plt.title("loss-epoch")
    plt.legend()
    plt.savefig(dir_path+"/mnist_loss_curve.png")

    plt.cla()
    plt.plot(range(len(train_acc_list)), train_acc_list, label="train")
    plt.plot(range(len(test_acc_list)), test_acc_list, label="test")
    plt.title("accuracy-epoch")
    plt.legend()
    plt.savefig(dir_path+"/mnist_accuracy_curve.png")


if __name__ == "__main__":
    net = build(decomp=False)
    get_num_params(net, "d FC net")
    #print(net)
    # print(net)
    # bp_train(20, net)
    es_train(30, net)
