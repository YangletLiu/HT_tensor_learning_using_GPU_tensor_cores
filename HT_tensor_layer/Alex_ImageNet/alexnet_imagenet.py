import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import torch.cuda.amp

import torchvision
import torchvision.transforms as transforms
from torchvision import models as models

import  matplotlib.pyplot as plt
# import tensorly as tl
# import tensorly
from itertools import chain
# from tensorly.decomposition import parafac, partial_tucker, tucker
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import time
from torch.autograd import Variable
from torch.distributions.multivariate_normal import MultivariateNormal

import torch.multiprocessing as mp
from torch.multiprocessing import Pool, Manager

import torchvision.datasets as datasets

from model_list import alexnet


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='alexnet',
                    help='model architecture (default: alexnet)')

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False


def data_loader(root, batch_size=256, workers=1, pin_memory=True):
    traindir = os.path.join(root, 'train')
    valdir = os.path.join(root, 'val')
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # normalize
        ])
    )
    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize((224,224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # normalize
        ])
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
        sampler=None
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory
    )
    return train_loader, val_loader


trainloader, val_loader = data_loader('/xfs/imagenet/',256, 0, True)
####################### ImageNet ##############################


cnttt = 0
 

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def weight_init(m):
    if isinstance(m,nn.Conv2d):
        nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class Optimizer():
    def __init__(self, lr, momentum, step_size, gamma) -> None:
        self.lr = lr
        self.momentum = momentum
        self.step_size = step_size
        self.gamma = gamma
        self.step = 0
    
    def step(self, gamma):
        self.step += 1


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():    
        for i, (input, target) in enumerate(val_loader):
            target = target.to(device)
            input = input.to(device)
            input_var = input
            target_var = target
            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target_var, topk=(1, 5))
            losses.update(loss.item(), input_var.size(0))
            top1.update(prec1[0], input_var.size(0))
            top5.update(prec5[0], input_var.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg


############################ functions for ES ######################################
def gen_noises(model,  layer_ids, std=1, co_matrices=None):
    noises = []
    for i, param in enumerate(model.parameters()):
        if i in layer_ids:
            if co_matrices == None:
                noises.append(torch.randn_like(param) * std)
            else:
                sz = co_matrices[i].shape[0]
                m = MultivariateNormal(torch.zeros(sz), co_matrices[i])
                noise = m.sample()
                noises.append(noise.reshape(param.shape))
        else:
            noises.append(torch.zeros_like(param))
        noises[-1] = noises[-1].to(device)
    return noises


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


def clone_params(model):
    params = []
    for param in model.parameters():
        params.append(torch.clone(param))
    return params


def es_update(model, epsilons, ls, lr, layer_ids, mode=1, update=True):
    device = epsilons[0][0].device
    num_directions = len(epsilons)
    elite_rate = 0.2
    elite_num = max(int(elite_rate * num_directions), 1)

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


    return grad
def set_es_grad(model, grad, lr, layer_ids, mode=1, update=True):
    if update:
        i = 0
        for g, param in zip(grad, model.parameters()):
            if i in layer_ids:
                param.requires_grad = False
                param -= lr * g
                param.requires_grad = True
            i += 1


def cma_es_update(model, epsilons, ls, layer_ids, params, mode=1, update=True):
    alpha_miu, alpha_sigma, alpha_cp, alpha_c1, alpha_clambda = params[:5]
    damping_factor = params[5]
    ps_sigma = params[6]
    co_matrices = params[7]
    sigmas = params[8]
    ps_c = sigmas[9]

    device = epsilons[0][0].device
    num_directions = len(epsilons)
    elite_rate = 0.2
    elite_num = max(int(elite_rate * num_directions), 1)

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
    delta = []
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
                    param -= alpha_miu * g
                    delta.append(-alpha_miu * g)
                    param.requires_grad = True
                else:
                    delta.append(g)
                i += 1
        else:
            i = 0
            # print(len(grad), layer_ids)
            for g, param in zip(grad, model.parameters()):
                if i in layer_ids:
                    # print("update")
                    param.requires_grad = False
                    param += alpha_miu * g
                    delta.append(alpha_miu * g)
                    param.requires_grad = True
                else:
                    delta.append(g)
                i += 1

    for i, g in enumerate(grad):
        if i in layer_ids:
            ps_sigma[i] = (1 - alpha_sigma) * ps_sigma[i] + torch.sqrt(alpha_sigma * (2 - alpha_sigma) * elite_num) * (1 / torch.sqrt(co_matrices[i])) * (delta[i] / sigmas[i])
            sigmas[i] = sigmas[i] * torch.exp(alpha_sigma / damping_factor * (torch.sqrt(torch.square(ps_sigma[i]).sum()) / np.sqrt(co_matrices[i].shape[0]) - 1))
            ps_c[i] = (1 - alpha_cp) * ps_c[i] + torch.sqrt(alpha_cp * (2 - alpha_cp) * elite_num) * (delta / sigmas[i])
            temp = 0
            for idx in indices:
                temp += torch.matmul(epsilons[idx][i], epsilons[idx][i].T)
            co_matrices[i] = (1 - alpha_clambda - alpha_c1) * co_matrices[i] + alpha_c1 * torch.matmul(ps_c[i], ps_c[i].T) + (1 / elite_num) * alpha_clambda * temp
    return grad, (ps_sigma, sigmas, ps_c, co_matrices)


def explore_one_direction(model, inputs, targets, criterion, layer_ids, co_matrices, if_mirror):
    ep_rt = []
    ls_rt = []

    epsilon = gen_noises(model, layer_ids, std=0.01, co_matrices=co_matrices)
    add_noises(model, epsilon, layer_ids)
    # with torch.cuda.amp.autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets).item()
    remove_noises(model, epsilon, layer_ids)

    ep_rt.append(epsilon.copy())
    ls_rt.append(loss)

    if if_mirror:        
        for i in range(len(epsilon)):
            epsilon[i] = -epsilon[i]
        add_noises(model, epsilon, layer_ids)
        # with torch.cuda.amp.autocast():
        outputs = model(inputs)
        loss = criterion(outputs, targets).item()
        remove_noises(model, epsilon, layer_ids)
        # ep_rt.append(epsilon)
        # ls_rt.append(loss)
        ls_rt[0] -= loss

    return ep_rt, ls_rt



def es_train(num_epoch, model):
    lr = 0.05
    lr0 = lr
    step_size = 20
    gamma = 0.5
    co_matrices = None
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    criterion = nn.CrossEntropyLoss()
    num_layers = len(model.state_dict())

    train_acc = []
    test_acc = []
    train_loss_list = []
    test_loss_list = []
    best_acc = 0
    # global_mean = []
    # global_var = []
    original_time = time.asctime(time.localtime(time.time()))

    model.train()
    model = model.to(device)
    model.share_memory()
    early_break = False


    es_mode = 1
    num_directions = 150
    num_directions0 = num_directions


    if_alternate = False
    if_es = True
    if_bp = False
    if_mirror = True
    # result = manager.list()
    start_time = time.time()

    try:
        if True:
            # for layer_id in range(num_layers):
            for ___ in range(1):
                layer_ids = list(range(num_layers))
                # layer_ids = [layer_id]
                num_directions = num_directions0
                if if_mirror:
                    num_directions = num_directions // 2
                lr = lr0
                # layer_ids = "alternate"
                for epoch in range(num_epoch):
                    # print("\nES layer ", "alternate" if if_alternate else layer_ids, "  Epoch: {}".format(epoch))
                    print("\n Epoch: {}".format(epoch))
                    print("|", end="")

                    losses = AverageMeter()
                    top1 = AverageMeter()
                    top5 = AverageMeter()

                    train_loss = 0
                    correct = 0
                    total = 0            
                    # epoch_mean = []
                    # epoch_var = []
                    model.train()
                    # lr = scheduler.get_lr()[0]
                    for batch_idx, (inputs, targets) in enumerate(trainloader):
                        # print("====================== layer ", layer_ids, " ======================")
                        inputs, targets = inputs.to(device), targets.to(device)
                        if if_alternate:
                            layer_ids = [layer_id]
                            layer_id = (layer_id + 1) % num_layers
                        # pool = Pool(num_directions)
                        total += len(inputs)
                        ls = []
                        epsilons = []

                        for _ in range(num_directions):
                            epsilon, loss = explore_one_direction(model, inputs, targets, criterion, layer_ids, co_matrices, if_mirror)
                            epsilons.extend(epsilon)
                            ls.extend(loss)
                            for l in loss:
                                train_loss += l

                        es_grad = es_update(model, epsilons, ls, lr, layer_ids, es_mode, update=if_es)
                        set_es_grad(model, es_grad, lr, layer_ids, es_mode, update=if_es)

                        # with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                        forward_loss = criterion(outputs, targets)

                        prec1, prec5 = accuracy(outputs.data, targets, topk=(1, 5))
                        loss = criterion(outputs, targets)
                        losses.update(forward_loss.item(), inputs.size(0))
                        top1.update(prec1[0], inputs.size(0))
                        top5.update(prec5[0], inputs.size(0))

                        if batch_idx % 10 == 0:
                            # print('=', end='')
                             print('Epoch: [{0}][{1}/{2}]\t'
                                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                                'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                                epoch, batch_idx, len(trainloader), 
                                loss=losses, top1=top1, top5=top5))

                    train_loss_list.append(train_loss)
                    train_acc.append(correct / total)
                    model.eval()
                    # best_acc = test(test_acc, test_loss_list, best_acc, model)
                    prec1 = validate(val_loader, model, criterion)

                    print('Current learning rate: ', lr)
                    print('Current num_directions: ', num_directions, "mirror" if if_mirror else "")
                    now_time = time.time()
                    print("used: {}  est: {}".format(now_time - start_time, (now_time - start_time) / (epoch + 1) * (num_epoch - epoch - 1)))
    except KeyboardInterrupt:
        early_break = True

    print('Best training accuracy overall: ', best_acc)
    
    
    return train_acc, test_acc, best_acc

#####################################################################################################



if __name__ == "__main__":
    args = parser.parse_args()
    if args.arch == 'alexnet':  # conventional layer
        model = models.alexnet(pretrained=True)
        model = model.to(device)

    elif args.arch == 'ht':  # HT tensor layer
        model = alexnet.ht(pretrained=args.pretrained)
        print('AlexNet HT model')        
        # input_size = 224
    else:
        raise Exception('Model not supported yet')

    es_train(30, model)




