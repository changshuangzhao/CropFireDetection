import os
import sys
import time
import torch
import argparse
import shutil
import torch.optim as optim
import torch.utils.data as data
from torch import nn
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR, MultiStepLR

sys.path.append(os.path.join(os.path.dirname(__file__), '../configs'))
from config import cfg
sys.path.append(os.path.join(os.path.dirname(__file__), '../prepare_data'))
from factory import dataloader, FireData
sys.path.append(os.path.join(os.path.dirname(__file__), '../networks'))
from resnet import resnet50, resnet18
from mobilenet import mobilenet_v2
from shufflenet import shufflenet_v2_x1_0
from DIY import DIYNet

sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))
from util import setup_logger, AverageMeter, accuracy


def params():
    parser = argparse.ArgumentParser(description='Fire Detector Training With Pytorch')
    parser.add_argument('--dataset', default='Foggia',
                        help='Train target')
    parser.add_argument('--number', '-n', default=3, type=int,
                        help='GPU ID')
    parser.add_argument('--batch_size', '-b', default=32, type=int,
                        help='Batch size for training')
    parser.add_argument('--resume', default=None, type=str,
                        help='Checkpoint state_dict file to resume training from')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers used in dataloading')
    parser.add_argument('--cuda', default=True, type=bool,
                        help='Use CUDA to train model')
    parser.add_argument('--lr', default=1e-5, type=float,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='Weight decay for optim')
    parser.add_argument('--multigpu', default=False, type=bool,
                        help='Use mutil Gpu training')
    parser.add_argument('--save_folder', default='weights/',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--log_dir', default='../logs',
                        help='Directory for saving logs')
    parser.add_argument('--print_freq', '-p', default=10, type=int,
                        help='print frequency (default: 10)')
    return parser.parse_args()


def main():
    args = params()
    writer = SummaryWriter(log_dir='run/')
    best_prec1 = 0

    # setup log path
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    logger = setup_logger(args.log_dir + '/training.log')

    # load train/test dataset
    train_images, train_labels = dataloader(os.path.join(cfg.Imgdir, 'TrainData'))
    test_images, test_labels = dataloader(os.path.join(cfg.Imgdir, 'TestData'))
    train_dataset = FireData(train_images, train_labels, True)
    test_dataset = FireData(test_images, test_labels, False)
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=args.batch_size,
                                   num_workers=args.num_workers,
                                   shuffle=True,
                                   pin_memory=True,
                                   drop_last=True)
    test_loader = data.DataLoader(test_dataset,
                                  batch_size=args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=False,
                                  pin_memory=True,
                                  drop_last=False)

    # setup cuda
    device = torch.device('cuda:{}'.format(args.number)) if args.cuda and torch.cuda.is_available() else 'cpu'
    # model
    # net = resnet18(pretrained=True, num_classes=2).to(device)
    net = mobilenet_v2(pretrained=True, num_classes=2).to(device)
    # net = shufflenet_v2_x1_0(pretrained=True, num_classes=2).to(device)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("==>loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)
            net.load_state_dict(checkpoint['state_dict'])
            print("==>loaded checkpoint '{}'".format(args.resume))
        else:
            print("==>no checkpoint found at '{}'".format(args.resume))
    cudnn.benckmark = True

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=args.weight_decay)
    optimizer = optim.RMSprop(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
    scheduler = MultiStepLR(optimizer, milestones=[20, 40], gamma=0.1)

    # train/test model
    for epoch in range(0, cfg.EPOCHS):
        train(train_loader, net, criterion, optimizer, epoch, args, logger, writer)
        prec1 = test(test_loader, net, criterion, args, epoch, logger, writer)
        scheduler.step()

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, epoch, args)
    logger.info(' ####################### Best accuracy:{} ######################## '.format(best_prec1))
    writer.close()


def save_checkpoint(state, is_best, epoch, args, filename='checkpoint.pth'):
    """Saves checkpoint to disk"""
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    filename = args.save_folder + str(epoch) + '_' + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, args.save_folder + 'model_best.pth')


def train(train_loader, net, criterion, optimizer, epoch, args, logger, writer):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to network mode
    net.train()

    end = time.time()
    for batch_idx, (images, labels) in enumerate(train_loader):
        if args.cuda and torch.cuda.is_available():
            images = images.cuda(args.number)
            labels = labels.cuda(args.number)
        output = net(images)
        loss = criterion(output, labels)

        # messure accuracy and record loss
        losses.update(loss.item(), images.size(0))

        prec1 = accuracy(output.data, labels, topk=(1,))[0]
        top1.update(prec1.item(), images.size(0))

        # compute gradient
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # logging
        itertion = batch_idx * epoch + batch_idx
        if itertion % args.print_freq == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                            epoch, batch_idx, len(train_loader),
                            batch_time=batch_time,
                            loss=losses,
                            top1=top1))

        # tensorboard
        writer.add_scalar('train_loss', losses.avg, epoch)
        writer.add_scalar('train_acc', top1.avg, epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)


def test(test_loader, net, criterion, args, epoch, logger, writer):
    """Perform test on the test set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to network mode
    net.eval()

    end = time.time()
    for batch_idx, (images, labels) in enumerate(test_loader):
        if args.cuda and torch.cuda.is_available():
            images = images.cuda(args.number)
            labels = labels.cuda(args.number)
        with torch.no_grad():
            output = net(images)
        loss = criterion(output, labels)

        # measure record loss, accuracy and elapsed time
        losses.update(loss.item(), images.size(0))
        prec1 = accuracy(output.data, labels, topk=(1,))[0]
        top1.update(prec1.item(), images.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % args.print_freq == 0:
            logger.info('Test: [{0}/{1}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                            batch_idx, len(test_loader),
                            batch_time=batch_time,
                            loss=losses,
                            top1=top1))
    logger.info(' ************************* [{Epoch}] Average Test Prec@1 {top1.avg:.3f} ********************** '.
                format(Epoch=epoch, top1=top1))

    # tensorboard
    writer.add_scalar('test_loss', losses.avg, epoch)
    writer.add_scalar('test_acc', top1.avg, epoch)

    return top1.avg


if __name__ == '__main__':
    main()
