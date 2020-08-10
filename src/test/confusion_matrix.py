import os
import sys
import torch
import argparse
import cv2 as cv
import numpy as np
from PIL import Image
from torchvision import transforms
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import itertools
import time


sys.path.append(os.path.join(os.path.dirname(__file__), '../networks'))
from resnet import resnet18
from mobilenet import mobilenet_v2
from shufflenet import shufflenet_v2_x1_0
from DIY import DIYNet
sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))
from util import accuracy, AverageMeter
sys.path.append(os.path.join(os.path.dirname(__file__), '../configs'))
from config import cfg
sys.path.append(os.path.join(os.path.dirname(__file__), '../prepare_data'))
from factory import dataloader


# imagenet_stats = {'mean':[0.0, 0.0, 0.0], 'std': [1.0, 1.0, 1.0]}
imagenet_stats = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}


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
    parser.add_argument('--lr', default=1e-4, type=float,
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


def demo():
    args = params()
    # net = resnet18(pretrained=False, num_classes=2)
    net = mobilenet_v2(pretrained=False, num_classes=2)
    # net = shufflenet_v2_x1_0(pretrained=False, num_classes=2)
    # net = DIYNet(num_classes=2)
    args.resume = os.path.join(os.path.dirname(__file__), '../train/weights/98.170model_best.pth')

    if os.path.isfile(args.resume):
        print("==>loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)
        net.load_state_dict(checkpoint['state_dict'])
        print("==>loaded checkpoint '{}'".format(args.resume))
    else:
        print("==>no checkpoint found at '{}'".format(args.resume))

    test_images, test_labels = dataloader('/Users/yanyan/data/CropFireData/TestData')
    atual_labels = np.array(test_labels)
    pre_labels = []

    processed = transforms.Compose([transforms.Resize((cfg.InputSize_h, cfg.InputSize_w)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(**imagenet_stats)])
    top1 = AverageMeter()
    times = AverageMeter()
    for test_image, test_label in zip(test_images, test_labels):
        labels = torch.tensor([test_label])
        labels = torch.unsqueeze(labels, dim=0)

        img = Image.open(test_image).convert('RGB')
        img = processed(img)
        img = torch.unsqueeze(img, dim=0)

        net.eval()

        with torch.no_grad():
            start = time.time()
            output = net(img)
            end = time.time()
        times.update(end - start)

        if torch.argmax(output).item() != labels[0][0].item():
            print('预测错误的图像：', test_image)
        pre_labels.append(torch.argmax(output).item())

        prec1 = accuracy(output.data, labels, topk=(1,))[0]
        top1.update(prec1.item(), img.size(0))
        # print(top1.val, top1.avg)

    pre_labels = np.array(pre_labels)
    cm = confusion_matrix(atual_labels, pre_labels)
    cm_plot_labels = ['Fire', 'No Fire']
    plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix')
    tp = cm[0][0]
    fn = cm[0][1]
    fp = cm[1][0]
    tn = cm[1][1]
    print('time avg: {:.4f}s'.format(end - start))
    print("tp" + ' ' + str(tp))
    print("fn" + ' ' + str(fn))
    print("fp" + ' ' + str(fp))
    print("tn" + ' ' + str(tn))
    acc = (tp + tn) / (tp + fn + fp + tn) * 100
    print('Accuracy: {:.2f}%'.format(acc))
    pre = tp / (tp + fp)
    print('Preision: {:.2f}%'.format(pre  * 100))
    rec = tp / (tp + fn)
    print('Recall: {:.2f}%'.format(rec * 100))
    print('F1 Score: {:.4f}'.format(2 * pre * rec / (pre + rec)))


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    demo()