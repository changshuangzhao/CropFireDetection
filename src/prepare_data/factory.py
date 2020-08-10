import os
import sys
from torch.utils import data
from PIL import Image
from torchvision import transforms
sys.path.append(os.path.join(os.path.dirname(__file__), '../configs'))
from config import cfg
import warnings
warnings.filterwarnings("error", category=UserWarning)


# imagenet_stats = {'mean':[0.0, 0.0, 0.0], 'std': [1.0, 1.0, 1.0]}
imagenet_stats = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}


def dataloader(filepath):
    images = []
    labels = []
    classes = ['Fire', 'NoFire']

    for tmp in classes:
        path = os.path.join(filepath, tmp)
        imgs_name = os.listdir(path)
        for img_name in imgs_name:
            if '.DS_Store' in img_name:
                continue
            img_path = os.path.join(path, img_name)
            images.append(img_path)
            index = classes.index(tmp)
            labels.append(index)
    return images, labels


class FireData(data.Dataset):
    def __init__(self, images, labels, train=True):
        self.images = images
        self.labels = labels
        self.training = train

    def __getitem__(self, index):
        try:
            image = Image.open(self.images[index]).convert('RGB')
        except UserWarning:
            print(self.images[index])
        label = self.labels[index]

        if self.training:
            processed = transforms.Compose([transforms.Resize((cfg.InputSize_h, cfg.InputSize_w)),
                                            transforms.RandomHorizontalFlip(0.5),
                                            transforms.RandomRotation(15),
                                            transforms.ToTensor(),
                                            transforms.Normalize(**imagenet_stats)])
        else:
            processed = transforms.Compose([transforms.Resize((cfg.InputSize_h, cfg.InputSize_w)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(**imagenet_stats)])

        image = processed(image)
        return image, label

    def __len__(self):
        return len(self.labels)


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    train_data = os.path.join(cfg.Imgdir, 'TrainData')
    images, labels = dataloader(train_data)
    data = DataLoader(FireData(images, labels), batch_size=1)
    for image, label in data:
        # img = np.array(image[0, :, :, :].permute(1, 2, 0))
        # plt.imshow(img)
        # plt.show()
        pass
    print('fishing')


