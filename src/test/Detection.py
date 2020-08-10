import os
import sys
import torch
import argparse
import cv2
import numpy as np
import time
from torchvision import transforms
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__)))
import frame2diff
sys.path.append(os.path.join(os.path.dirname(__file__), '../networks'))
from resnet import resnet18
from mobilenet import mobilenet_v2
sys.path.append(os.path.join(os.path.dirname(__file__), '../configs'))
from config import cfg
sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))


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
    # option video/image/image group
    option = 'image group'
    args = params()
    # net = resnet18(pretrained=False, num_classes=2)
    net = mobilenet_v2(pretrained=False, num_classes=2)
    args.resume = os.path.join(os.path.dirname(__file__), '../train/weights/98.170model_best.pth')

    if os.path.isfile(args.resume):
        print("==>loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)
        net.load_state_dict(checkpoint['state_dict'])
        print("==>loaded checkpoint '{}'".format(args.resume))
    else:
        print("==>no checkpoint found at '{}'".format(args.resume))

    processed = transforms.Compose([transforms.Resize((cfg.InputSize_h, cfg.InputSize_w)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(**imagenet_stats)])
    if option == 'video':
        cap = cv2.VideoCapture('/Users/yanyan/Downloads/FireNet-LightWeight-Network-for-Fire-Detection-master/Raspberry_Pi_3B_Sample_Results/Result_2.mkv')
        while cap.isOpened():
            ret, image = cap.read()
            if ret:
                orig = image.copy()
                boxes = frame2diff.check_color(image)
                fire_boxes = []
                for box in boxes:
                    detection_img = image[box[1]:box[3] + 1, box[0]:box[2] + 1]
                    detection_img = Image.fromarray(cv2.cvtColor(detection_img, cv2.COLOR_BGR2RGB))
                    detection_img = processed(detection_img)
                    detection_img = torch.unsqueeze(detection_img, dim=0)

                    net.eval()
                    with torch.no_grad():
                        output = net(detection_img)
                    fire_prob = torch.nn.Softmax(dim=1)(output).numpy()[0][0] * 100
                    if fire_prob >= 50:
                        fire_boxes.append(box)
                if fire_boxes:
                    for fire_box in fire_boxes:
                        x1, y1, x2, y2 = fire_box
                        cv2.rectangle(orig, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.imshow('fire detection', orig)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        cap.release()
        cv2.destroyAllWindows()

    elif option == 'image':
        image_path = '/Users/yanyan/data/FireData/TestData/Fire/fire2278.jpg'
        img1 = cv2.imread(image_path)
        # print(img1)
        # orig = img1.copy()
        boxes = frame2diff.check_color(img1)

        fire_boxes = []
        for box in boxes:
            detection_img = img1[box[1]:box[3] + 1, box[0]:box[2] + 1]
            img = Image.fromarray(cv2.cvtColor(detection_img, cv2.COLOR_BGR2RGB))
            img = processed(img)
            img = torch.unsqueeze(img, dim=0)

            net.eval()
            with torch.no_grad():
                tic = time.time()
                output = net(img)
                toc = time.time()
            fire_prob = torch.nn.Softmax(dim=1)(output).numpy()[0][0] * 100
            if fire_prob >= 50:
                fire_boxes.append(box)
            print("Time taken = ", toc - tic)
            print("FPS: ", 1 / np.float64(toc - tic))
            print('Fire Probability', fire_prob)

        if fire_boxes:
            for fire_box in fire_boxes:
                x1, y1, x2, y2 = fire_box
                cv2.rectangle(img1, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.imshow('fire detection', img1)
            cv2.waitKey()

    elif option == 'image group':
        # root = '/Users/yanyan/Downloads/ImageDownload_fire/燃烧的森林'
        root = '/Users/yanyan/data/FireData/TestData/Fire'
        # root = '/Users/yanyan/Desktop/fireandsmoke/normal'
        img_names = os.listdir(root)
        for img_name in img_names:
            if '.DS_Store' in img_name:
                continue
            image_path = os.path.join(root, img_name)
            print(image_path)
            img = cv2.imread(image_path)
            orig = img.copy()
            boxes = frame2diff.check_color(img)
            fire_boxes = []
            for box in boxes:
                detection_img = img[box[1]:box[3] + 1, box[0]:box[2] + 1]
                detection_img = Image.fromarray(cv2.cvtColor(detection_img, cv2.COLOR_BGR2RGB))
                detection_img = processed(detection_img)
                detection_img = torch.unsqueeze(detection_img, dim=0)

                net.eval()
                with torch.no_grad():
                    # st = time.time()
                    output = net(detection_img)
                    # print(time.time() - st)
                fire_prob = torch.nn.Softmax(dim=1)(output).numpy()[0][0] * 100
                if fire_prob > 50:
                    fire_boxes.append(box)

            if fire_boxes:
                for fire_box in fire_boxes:
                    x1, y1, x2, y2 = fire_box
                    cv2.rectangle(orig, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.imshow('fire detection', orig)

            if cv2.waitKey() == 27:
                exit()


if __name__ == '__main__':
    demo()
