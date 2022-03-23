import argparse
import os
import shutil
import time

# from apex import amp
import torch.cuda.amp

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt

from model_list import alexnet
from torchsummary import summary
import sys
import gc
cwd = os.getcwd()
sys.path.append(cwd+'/../')




# set the seed
torch.manual_seed(1)
torch.cuda.manual_seed(1)
device = 'cuda:7' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='alexnet',
                    help='model architecture (default: alexnet)')
# parser.add_argument('--data', metavar='DATA_PATH', default='./data/',
#                     help='path to imagenet data (default: ./data/)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=5, type=int, metavar='N', #50   epoch
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
# parser.add_argument('-b', '--batch-size', default=256, type=int, #batch_size (默认256)
#                     metavar='N', help='mini-batch size (default: 128)')
# parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
#                     metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.90, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='checkpoint_ht.pth.tar', type=str, metavar='PATH', #模型路径
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set',default=False)
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    default=False, help='use pre-trained model')


best_prec1 = 0
batch_size = 256
std = 0.025
lr0 = 0.01

################   2. load data   #####################################
def data_loader(root, batch_size=256, workers=1, pin_memory=True):
    traindir = os.path.join(root, 'train')
    valdir = os.path.join(root, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    )
    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize((224,224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
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



############### 3. function on train ###############################


def save_checkpoint(state, is_best, filename='checkpoint_ht.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best_ht.pth.tar')


def adjust_learning_rate(optimizer, epoch,init_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 40 epochs"""
    lr = init_lr * (0.1 ** (epoch // 15)) # 40 --> 15
    print('Learning rate:', lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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



############# 4. validate   #############
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

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
    with open("validate.txt", "a") as f:
        f.write(str(top1.val)+'_'+str(top1.avg)+'\n')
    with open("validate_loss.txt", "a") as f:
        f.write(str(losses.val)+'_'+str(losses.avg)+'\n')

    return top1.avg

############################# 5.  ES function ##################
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


def explore_one_direction(net,criterion, data, if_mirror):
    inputs, targets = data
    noise = gen_noise(net)

    add_noise(net, noise)

    with torch.cuda.amp.autocast():
        outputs = net(inputs)
    # outputs = net(inputs)

    remove_noise(net, noise)
    loss = criterion(outputs, targets).item()
    if if_mirror:
        inverse_noise = []
        for i in range(len(noise)):
            inverse_noise.append(-noise[i])
        add_noise(net, inverse_noise)

        with torch.cuda.amp.autocast():
            outputs = net(inputs)
        # outputs = net(inputs)

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

###################### 6. train_es ###################

def train_ES(train_loader, model, criterion, optimizer, epoch,batch_size,if_mirror,num_directions,elite_num):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()



    # switch to train mode
    model.train()
    model.training = True

    end = time.time()
    with torch.no_grad():
        for i, (inputs, target) in enumerate(train_loader):
            # measure data loading time
            start_time = time.time()
            data_time.update(time.time() - end)

            inputs, targets = inputs.to(device), target.to(device)


            # compute output with tensor core
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
            # outputs = model(inputs) 
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets, topk=(1, 5))

            # updata loss and accuracy
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1[0], inputs.size(0))
            top5.update(prec5[0], inputs.size(0))

            # update parameters
            optimizer.zero_grad()
            loss_list = []
            noise_list = []
            for _ in range(num_directions):
                l, noise = explore_one_direction(model,criterion, (inputs, targets), if_mirror=if_mirror)
                loss_list.append(l)
                noise_list.append(noise)
            # grad = get_es_grad(model,loss_list,batch_size, noise_list, elite_num)
            grad = get_es_grad(loss_list, noise_list, elite_num)
            set_es_grad(model, grad)

            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end_time = time.time()

            if i % args.print_freq == 0:
                if(epoch == 1 or epoch == 2):
                    para = list(model.named_parameters())
                    tmp = input('--- wait ---')
                    print(para)

                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       epoch, i, len(train_loader), batch_time=batch_time,
                       data_time=data_time, loss=losses, top1=top1, top5=top5))
                print("10 batch cost time :",end_time - start_time)
            gc.collect()

    # save loss and accuracy
    with open("train_acc.txt", "a") as f:
        f.write(str(top1.val)+'_'+str(top1.avg)+'\n')
    with open("loss.txt", "a") as f:
        f.write(str(losses.val)+'_'+str(losses.avg)+'\n')

def es_train(model,train_loader,val_loader):

    if_mirror = True
    num_directions = 150 # natural gradient iteration number
    elite_rate = 1.
    elite_num = max(int(elite_rate * num_directions), 1)
    model.to(device)
    global best_prec1
    # current_lr = lr0

    for epoch in range(0, args.epochs):
        adjust_learning_rate(optimizer, epoch,lr0)
        
        train_ES(train_loader, model, criterion, optimizer, epoch, batch_size,if_mirror,num_directions,elite_num)

        prec1 = validate(val_loader, model, criterion)

        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)




if __name__ == "__main__":
    args = parser.parse_args()
    if args.arch=='alexnet': # conventional layer 
        model = alexnet.alexnet(pretrained=args.pretrained)
        # input_size = 224
    elif args.arch=='ht': # HT tensor layer
        model = alexnet.ht(pretrained=args.pretrained)
        print('AlexNet HT model')
        # input_size = 224
    else:
        raise Exception('Model not supported yet')
    

    # cudnn.benchmark = True
    pin_memory = True
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr0,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    train_loader, val_loader = data_loader('/xfs/imagenet/',batch_size, args.workers, pin_memory)
    es_train(model,train_loader,val_loader)  