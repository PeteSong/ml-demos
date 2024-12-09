import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms
from PIL import Image
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchvision.models import VGG16_Weights
from torchvision.models import vgg16

from utils import get_device, show_image, load_saved_model, save_model


def load_and_process_image(img_path, pre_transforms, device, enhance_transforms=None, show_img=False):
    if show_img:
        show_image(img_path)
    with Image.open(img_path) as img:
        # transform image data to fit the model
        img_tensor = pre_transforms(img).to(device)
        # enhance data
        if enhance_transforms:
            img_tensor = enhance_transforms(img_tensor)
    return img_tensor


def load_vgg_model(device):
    weights = VGG16_Weights.DEFAULT
    model = vgg16(weights=weights)
    return model


def load_vgg_transforms():
    weights = VGG16_Weights.DEFAULT
    trans = weights.transforms()
    return trans


def init_model(device):
    vgg_mdl, pre_trans = load_vgg_model(device), load_vgg_transforms()
    vgg_mdl.requires_grad_(False)
    print('VGG16 Frozen')
    N_CLASSES = 1
    my_model = nn.Sequential(
        vgg_mdl,
        nn.Linear(1000, N_CLASSES)
    )
    my_model.to(device)
    # check the parameters
    # for idx, param in enumerate(my_model.parameters()):
    #     print(f'Parameter {idx} is frozen: {param.requires_grad}')
    return vgg_mdl, my_model, pre_trans


def get_batch_accuracy(output, y, N):
    zero_tensor = torch.tensor([0]).to(output.device)
    pred = torch.gt(output, zero_tensor)
    correct = pred.eq(y.view_as(pred)).sum().item()
    return correct / N


class MyDataset(Dataset):
    DATA_LABELS = ['bo', 'not_bo']
    IMG_SIZE = (224, 224)

    def __init__(self, data_dir, pre_transforms, device):
        self.img_paths = []
        self.labels = []
        self.pre_transforms = pre_transforms
        self.device = device
        # Determine the label per the path
        for l_idx, label in enumerate(self.DATA_LABELS):
            label_tensor = torch.tensor(l_idx).float().to(self.device)

            data_path = os.path.join(data_dir, label)
            data_paths = Path(data_path).rglob('*.jpg')
            for p in data_paths:
                self.img_paths.append(p)
                self.labels.append(label_tensor)
        self.transforms = transforms.Compose([
            transforms.RandomRotation(25),
            # In MAC, `antialias` is not implemented for MPS
            transforms.RandomResizedCrop(self.IMG_SIZE, scale=(.8, 1), ratio=(1, 1), antialias=False),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=.2, contrast=.2, saturation=.2, hue=.2)
        ])

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = load_and_process_image(img_path, self.pre_transforms, self.device, self.transforms)
        label = self.labels[idx]
        return img, label

    def __len__(self):
        return len(self.img_paths)


def load_presidential_dog_data(pre_transforms, device):
    n = 32
    train_path = './data/presidential_doggy_door/train/'
    train_data = MyDataset(train_path, pre_transforms, device)
    train_loader = DataLoader(train_data, batch_size=n, shuffle=True)

    valid_path = './data/presidential_doggy_door/valid/'
    valid_data = MyDataset(valid_path, pre_transforms, device)
    valid_loader = DataLoader(valid_data, batch_size=n)

    return train_loader, valid_loader


def load_vgg_classes():
    vgg_classes = json.load(open('./data/imagenet_class_index.json'))
    return vgg_classes


def train(model, train_loader, loss_func, optimizer):
    loss = 0
    accuracy = 0

    train_N = len(train_loader.dataset)
    device = next(model.parameters()).device

    model.train()
    c = 0
    for x, y in train_loader:
        # print(len(x))
        c += 1
        if x.device != device or y.device != device:
            x, y = x.to(device), y.to(device)
        output = torch.squeeze(model(x))
        optimizer.zero_grad()
        batch_loss = loss_func(output, y)
        batch_loss.backward()
        optimizer.step()

        loss += batch_loss.item()
        accuracy += get_batch_accuracy(output, y, train_N)
    print(f'Train => Loss:{loss:.4f}, Accuracy:{accuracy:.4f}, count: {c}')


def validate(model, valid_loader, loss_func):
    loss = 0
    accuracy = 0

    valid_N = len(valid_loader.dataset)
    device = next(model.parameters()).device

    model.eval()
    with torch.no_grad():
        for x, y in valid_loader:
            if x.device != device or y.device != device:
                x, y = x.to(device), y.to(device)
            output = torch.squeeze(model(x))

            loss += loss_func(output, y).item()
            accuracy += get_batch_accuracy(output, y, valid_N)
    print(f'Valid => Loss:{loss:.4f}, Accuracy:{accuracy:.4f}\n')


def predict(model, pre_transforms, img_path):
    dvc = next(model.parameters()).device
    img = load_and_process_image(img_path, pre_transforms, dvc, show_img=True)
    img = img.unsqueeze(0)
    output = model(img)
    prediction = output.item()
    print(prediction)
    return prediction


def presidential_doggy_door(model, pre_transforms, img_path):
    prediction = predict(model, pre_transforms, img_path)
    if prediction < 0:
        print("It is Bo! Let him in!\n")
    else:
        print('That is not Bo! Stay out!\n')


if __name__ == '__main__':
    VGG_PRESIDENTIAL_DOGGY_DOOR_MODEL_PATH = './saved_models/vgg_presidential_doggy_door_model.pth'
    dvc = get_device()

    if os.path.exists(VGG_PRESIDENTIAL_DOGGY_DOOR_MODEL_PATH):
        mdl = load_saved_model(VGG_PRESIDENTIAL_DOGGY_DOOR_MODEL_PATH)
        pre_trans = load_vgg_transforms()
    else:
        vgg_model, mdl, pre_trans = init_model(dvc)
        loss_function = nn.BCEWithLogitsLoss()
        optimizer = Adam(mdl.parameters())
        train_loader, valid_loader = load_presidential_dog_data(pre_trans, dvc)
        print('Started training with frozen VGG16.')
        for epoch in range(10):
            print('Epoch: {}'.format(epoch))
            train(mdl, train_loader, loss_function, optimizer)
            validate(mdl, valid_loader, loss_function)
        # predict(mdl, pre_trans, './data/presidential_doggy_door/valid/bo/bo_20.jpg')
        # predict(mdl, pre_trans, './data/presidential_doggy_door/valid/not_bo/121.jpg')
        ## after training, then we can tune the model based on the trained model.
        print('\nAfter training, tuning the trained model ...')
        print('Un-frozen VGG16.')
        vgg_model.requires_grad_(True)
        optimizer_for_tuning = Adam(mdl.parameters(), lr=.000001)
        for epoch in range(2):
            print('Epoch: {}'.format(epoch))
            train(mdl, train_loader, loss_function, optimizer_for_tuning)
            validate(mdl, valid_loader, loss_function)
        # predict(mdl, pre_trans, './data/presidential_doggy_door/valid/bo/bo_20.jpg')
        # predict(mdl, pre_trans, './data/presidential_doggy_door/valid/not_bo/121.jpg')
        save_model(mdl, VGG_PRESIDENTIAL_DOGGY_DOOR_MODEL_PATH)
    presidential_doggy_door(mdl, pre_trans, './data/presidential_doggy_door/valid/bo/bo_27.jpg')
    presidential_doggy_door(mdl, pre_trans, './data/presidential_doggy_door/valid/bo/bo_29.jpg')
    presidential_doggy_door(mdl, pre_trans, './data/presidential_doggy_door/valid/not_bo/126.jpg')
    presidential_doggy_door(mdl, pre_trans, './data/presidential_doggy_door/valid/not_bo/140.jpg')
