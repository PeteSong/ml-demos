import os
from pathlib import Path

import torch
import torchvision.io as tv_io
import torchvision.transforms.v2 as transforms
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision.models import VGG16_Weights, vgg16
from utils import get_device, load_saved_model, save_model, show_image, train_some_times


def load_and_process_image(img_path, pre_transforms, device, enhance_transforms=None, show_img=False):
    if show_img:
        show_image(img_path)
    img = tv_io.read_image(img_path, tv_io.ImageReadMode.RGB)
    img_tensor = pre_transforms(img).to(device)
    # with Image.open(img_path) as img:
    # transform image data to fit the model
    # img_tensor = pre_transforms(img).to(device)
    # enhance data
    if enhance_transforms:
        img_tensor = enhance_transforms(img_tensor)
    return img_tensor


def load_vgg_model():
    wgts = VGG16_Weights.DEFAULT
    mdl = vgg16(weights=wgts)
    return mdl


def load_vgg_transforms():
    wgts = VGG16_Weights.DEFAULT
    trans = wgts.transforms()
    return trans


def init_model(device):
    vgg_mdl = load_vgg_model()

    # VGG16 model Frozen
    vgg_mdl.requires_grad_(False)

    n_classes = 6
    my_model = nn.Sequential(
        vgg_mdl.features,
        vgg_mdl.avgpool,
        nn.Flatten(),
        vgg_mdl.classifier[0:3],
        nn.Linear(4096, 500),
        nn.ReLU(),
        nn.Linear(500, n_classes),
    )
    my_model.to(device)
    if torch.cuda.is_available():
        my_model.compile()
    return my_model


DATA_LABELS = ["freshapples", "freshbanana", "freshoranges", "rottenapples", "rottenbanana", "rottenoranges"]
IMG_SIZE = (224, 224)


class MyDataset(Dataset):
    def __init__(self, data_dir, pre_transforms, device):
        self.img_paths = []
        self.labels = []
        self.pre_transforms = pre_transforms
        self.device = device
        # Determine the label per the path
        for l_idx, label in enumerate(DATA_LABELS):
            label_tensor = torch.tensor(l_idx).float().to(self.device)

            data_path = os.path.join(data_dir, label)
            data_paths = Path(data_path).rglob('*.png')
            for p in data_paths:
                self.img_paths.append(p)
                self.labels.append(label_tensor)
        antialias = True
        # In MAC, `antialias` is not implemented for MPS
        if torch.backends.mps.is_available():
            antialias = False
        self.transforms = transforms.Compose(
            [
                transforms.RandomRotation(25),
                transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1), ratio=(1, 1), antialias=antialias),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ]
        )

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = load_and_process_image(img_path, self.pre_transforms, self.device, self.transforms)
        label = self.labels[idx]
        return img, label

    def __len__(self):
        return len(self.img_paths)


def load_fruits_data(pre_transforms, device):
    n = 32
    tr_path = '../data/fruits/train/'
    tr_data = MyDataset(tr_path, pre_transforms, device)
    tr_loader = DataLoader(tr_data, batch_size=n, shuffle=True)

    vld_path = '../data/fruits/valid/'
    vld_data = MyDataset(vld_path, pre_transforms, device)
    vld_loader = DataLoader(vld_data, batch_size=n)

    return tr_loader, vld_loader


def predict(model, pre_transforms, img_path):
    dvc = next(model.parameters()).device
    img = load_and_process_image(img_path, pre_transforms, dvc, show_img=True)
    # batch it
    img = img.unsqueeze(0)
    output = model(img)
    # print(output)
    prediction = output.argmax(dim=1).item()
    print(prediction)
    return prediction


if __name__ == '__main__':
    VGG_FRUITS_MODEL_PATH = '../saved_models/vgg_fruits_model.pth'
    dvc = get_device()
    pre_trans = load_vgg_transforms()
    train_loader, valid_loader = load_fruits_data(pre_trans, dvc)

    if os.path.exists(VGG_FRUITS_MODEL_PATH):
        model = load_saved_model(VGG_FRUITS_MODEL_PATH)
    else:
        model = init_model(dvc)
        loss_func = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters())
        train_some_times(model, train_loader, valid_loader, loss_func, optimizer, 10)
        save_model(model, VGG_FRUITS_MODEL_PATH)

    predict(model, pre_trans, '../data/fruits/valid/freshapples/Screen Shot 2018-06-08 at 5.01.15 PM.png')
    predict(model, pre_trans, '../data/fruits/valid/freshbanana/Screen Shot 2018-06-12 at 9.38.04 PM.png')
    predict(model, pre_trans, '../data/fruits/valid/freshoranges/Screen Shot 2018-06-12 at 11.50.41 PM.png')
    predict(model, pre_trans, '../data/fruits/valid/rottenapples/Screen Shot 2018-06-07 at 2.20.04 PM.png')
    predict(model, pre_trans, '../data/fruits/valid/rottenbanana/Screen Shot 2018-06-12 at 8.51.00 PM.png')
    predict(model, pre_trans, '../data/fruits/valid/rottenoranges/Screen Shot 2018-06-12 at 11.19.56 PM.png')
