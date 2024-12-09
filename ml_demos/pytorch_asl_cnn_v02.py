import pandas as pd
import torch
import torch.nn as nn
import torchvision.io as tv_io
import torchvision.transforms.v2 as transforms
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from utils import get_device, load_saved_model, save_model, show_image, train_some_times

IMG_HEIGHT = 28
IMG_WIDTH = 28
IMG_CHS = 1
N_CLASSES = 24


class MyDataset(Dataset):
    def __init__(self, base_df, device):
        x_df = base_df.copy()
        y_df = x_df.pop('label')
        x_df = x_df.to_numpy() / 255
        x_df = x_df.reshape(-1, IMG_CHS, IMG_WIDTH, IMG_HEIGHT)
        self.xs = torch.tensor(x_df).float().to(device)
        self.ys = torch.tensor(y_df).to(device)
        # 随机变化的序列
        self.trans = transforms.Compose(
            [
                transforms.RandomRotation(5),
                # In MAC, do not use `antialias` since it's not implemented in MPS yet.
                transforms.RandomResizedCrop((IMG_WIDTH, IMG_HEIGHT), scale=(0.9, 1), ratio=(1, 1), antialias=False),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.5),
            ]
        )

    def __getitem__(self, idx):
        x = self.xs[idx]
        y = self.ys[idx]

        # 用随机变换，增强数据
        x = self.trans(x)
        return x, y

    def __len__(self):
        return len(self.xs)


def load_data(device):
    train_df = pd.read_csv('../data/asl_data/sign_mnist_train.csv')
    valid_df = pd.read_csv('../data/asl_data/sign_mnist_valid.csv')

    BATCH_SIZE = 32
    train_data = MyDataset(train_df, device)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    valid_data = MyDataset(valid_df, device)
    valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=True)
    return train_loader, valid_loader


class MyConvolutionBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_p):
        kernel_size = 3
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.MaxPool2d(2, stride=2),
        )

    def forward(self, x):
        return self.model(x)


def init_model(device):
    flattened_img_size = 75 * 3 * 3

    # input 1 * 28 * 28
    base_model = nn.Sequential(
        MyConvolutionBlock(IMG_CHS, 25, 0),  # 25 * 14 * 14
        MyConvolutionBlock(25, 50, 0.2),  # 50 * 7 * 7
        MyConvolutionBlock(50, 75, 0),  # 75 * 3 * 3
        # Flatten to Dense layers
        nn.Flatten(),
        nn.Linear(flattened_img_size, 512),
        nn.Dropout(0.3),
        nn.ReLU(),
        nn.Linear(512, N_CLASSES),
    )

    base_model.to(device)
    if torch.cuda.is_available():
        base_model.compile()
    return base_model


def predict(model, arg, expected_output):
    device = next(model.parameters()).device
    if device != arg.device:
        arg = arg.to(device)
    prediction = model(arg)
    indices = prediction.argmax(dim=1)
    if expected_output.device != device:
        expected_output = expected_output.to(device)
    is_passed = indices.eq(expected_output)
    true_count = is_passed.sum().item()
    false_count = len(is_passed) - true_count
    print(
        f'''
    Indices of the maximum values in Prediction: {indices}
    expected output: {expected_output}
    Is passed: {is_passed}
    Passed Count - True: {true_count}, False: {false_count}
    '''
    )
    return indices


def predict_letter(model, file_path):
    show_image(file_path)
    device = next(model.parameters()).device

    image = tv_io.read_image(file_path, tv_io.ImageReadMode.GRAY)
    preprocess_trans = transforms.Compose(
        [
            transforms.ToDtype(torch.float32, scale=True),  # Converts [0, 255] to [0, 1]
            transforms.Resize((IMG_WIDTH, IMG_HEIGHT)),
            transforms.Grayscale(),  # From color to gray
        ]
    )
    image = preprocess_trans(image)
    image = image.unsqueeze(0).to(device)
    output = model(image)
    prediction = output.argmax(dim=1).item()
    alphabet = "abcdefghiklmnopqrstuvwxy"
    predicted_letter = alphabet[prediction]
    return predicted_letter


if __name__ == '__main__':
    ASL_NN_MODEL_PATH = '../saved_models/asl_nn_mode_v02.pth'
    device = get_device()
    train_loader, valid_loader = load_data(device)

    import os

    if os.path.exists(ASL_NN_MODEL_PATH):
        model = load_saved_model(ASL_NN_MODEL_PATH)
    else:
        model = init_model(device)
        loss_func = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters())
        train_some_times(model, train_loader, valid_loader, loss_func, optimizer, 10)
        save_model(model, ASL_NN_MODEL_PATH)
    batch = next(iter(valid_loader))
    predict(model, batch[0], batch[1])

    ifp = '../data/asl_images/b.png'
    letter = predict_letter(model, ifp)
    print(f'image: {ifp}, predicted letter is {letter}')
