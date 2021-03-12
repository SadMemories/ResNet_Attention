import argparse
import os
import random
import time
import shutil
from tqdm import tqdm

import torch
import torchvision
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from model import *
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='CIFAR10 Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet', help='model architesture')
parser.add_argument('--depth', default=50, type=int, help='model depth')
parser.add_argument('--device', default='cuda', type=str, help='index of gpu, i,e:0,1')
parser.add_argument('--epochs', default=60, type=int, help='number of total epochs to run')
parser.add_argument('--batch_size', default=128, type=int, help='mini-batch size')
parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint')
parser.add_argument('--att_type', default='None', choices=['cbam', 'se'], type=str)
parser.add_argument("--prefix", type=str, required=True, metavar='PFX', help='prefix for logging & checkpoint saving')
parser.add_argument('--syncBN', type=bool, default=True)
parser.add_argument('--world-size', default=2, type=int, help='number of distributed processes')
parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

seed = 1000  # Fixed random seed
lr = 0.1  # learning rate
print_freq = 10  # print frequency
writer = SummaryWriter()

if not os.path.exists('./checkpoints'):
    os.mkdir('./checkpoints')


def main():
    if torch.cuda.is_available() is False:
        raise EnvironmentError('not find CPU device for training.')

    args = parser.parse_args()
    print('args ', args)

    start_epoch = 0  # if args.resume is True, start_epoch will be set later
    best_prec1 = 0  # highest precision

    # use fixed random seed, you can reproduce the result
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

    device = select_device(args.device)

    if args.arch == 'resnet':

        # use CIFAR10 data set
        # print(args.att_type)
        model = ResidualNet('CIFAR10', args.depth, 10, args.att_type)

    # recommended to use DistributedDataParallel
    if torch.cuda.device_count() > 1:
        print("using {} gpus to train".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))

    model = model.to(device)
    print('model:')
    print(model)

    # add model architecture to graph
    tmp_input = torch.zeros([16, 3, 32, 32]).to(device)
    writer.add_graph(model, tmp_input)

    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.9, weight_decay=1e-4)

    # get the number of model parameters
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.exists(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # set benchmark to True, the training speed will be accelerated
    cudnn.benchmark = True

    num_workers = min([os.cpu_count(), args.batch_size if args.batch_size > 0 else 0, 8])

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))  # CIFAR10's mean and std
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))  # CIFAR10's mean and std
    ])

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=False, transform=train_transform)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=False, transform=test_transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,
                                               shuffle=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size,
                                              shuffle=False, num_workers=num_workers)

    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        loss, top1_acc, top5_acc = train(train_loader, model, criterion, optimizer, epoch, args.epochs, device)
        writer.add_scalar('train/epoch loss', loss, epoch+1)
        writer.add_scalar('train/top1 acc', top1_acc, epoch+1)
        writer.add_scalar('train/top5 acc', top5_acc, epoch+1)

        # evaluate on validation set
        prec1, prec5, val_loss = validate(test_loader, model, criterion, epoch, args.epochs, device)
        writer.add_scalar('val/loss', val_loss, epoch+1)
        writer.add_scalar('val/top1 acc', prec1, epoch + 1)
        writer.add_scalar('val/top5 acc', prec5, epoch + 1)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict()
        }, is_best, args.prefix)
    writer.close()


def select_device(device):
    cpu_request = (device == 'cpu')

    if device and not cpu_request:
        os.environ['CUDA_VISIBLE_DEVICES'] = device
        assert torch.cuda.is_available(), 'CUDA unavailable, invalid device requested'

    cuda = False if cpu_request else torch.cuda.is_available()

    return torch.device('cuda:0' if cuda else 'cpu')


def train(train_loader, model, criterion, optimizer, epoch, epochs, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    print(('\n' + '%10s' * 6) % ('Epoch', 'BatchTime', 'DataTime', 'Loss', 'Prec@1', 'Prec@5'))
    train_bar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (input, target) in train_bar:
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.to(device)
        input = input.to(device)

        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            s = ('%10s' * 1 + '%10.3g' * 5) % ('%g/%g' % (epoch+1, epochs), batch_time.avg, data_time.avg, losses.avg, top1.avg / 100.0, top5.avg / 100.0)
            # put s in front of train_bar
            train_bar.set_description(s)
            '''
            print('Epoch: [{0}/{1}][{2}/{3}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, epochs, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
            '''
    return losses.avg, top1.avg / 100.0, top5.avg / 100.0


def validate(test_loader, model, criterion, epoch, epochs, device):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    s = ('%30s' + '%10s' * 3) % ('BatchTime', 'Loss', 'Prec@1', 'Prec@5')
    test_bar = tqdm(test_loader, desc=s)
    for i, (input, target) in enumerate(test_bar):
        target = target.to(device)
        input = input.to(device)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        '''
        if i % print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i, len(test_loader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5))
        '''
    print(('%30.3g' + '%10.3g' * 3) % (batch_time.avg, losses.avg, top1.avg / 100.0, top5.avg / 100.0))
    '''
    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
    '''

    return top1.avg / 100.0, top5.avg / 100.0, losses.avg


def save_checkpoint(state, is_best, prefix):
    filename = './checkpoints/%s_checkpoint.pth.tar' % prefix
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, './checkpoints/%s_model_best.pth.tar' % prefix)


def accuracy(output, target, topk=(1,)):
    # compute the precision@k for the specified values of k
    maxk = max(topk)
    batch_size = target.size(0)  # target: (1, batch)

    # explain: https://blog.csdn.net/qq_34914551/article/details/103738160
    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)  # pred: (batch, maxk)
    pred = pred.t()  # pred: (maxk, batch)
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        # print(correct[:k].flatten().size())
        # print(correct[:k].size())
        correct_k = correct[:k].flatten().float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res


class AverageMeter():
    # compute and store the average and current value
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


def adjust_learning_rate(optimizer, epoch):
    # sets the learning rate to the initial LR decayed by 10 every 30 epochs
    current_lr = lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr


if __name__ == '__main__':
    main()