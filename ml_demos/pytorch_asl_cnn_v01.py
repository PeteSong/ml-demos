import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from utils import get_device, load_saved_model, save_model, train_some_times

IMG_HEIGHT = 28
IMG_WIDTH = 28
IMG_CHS = 1


class MyDataset(Dataset):
    def __init__(self, base_df, device):
        x_df = base_df.copy()
        y_df = x_df.pop('label')
        x_df = x_df.to_numpy() / 255
        x_df = x_df.reshape(-1, IMG_CHS, IMG_WIDTH, IMG_HEIGHT)
        self.xs = torch.tensor(x_df).float().to(device)
        self.ys = torch.tensor(y_df).to(device)

    def __getitem__(self, idx):
        x = self.xs[idx]
        y = self.ys[idx]
        return x, y

    def __len__(self):
        return len(self.xs)


def load_data(device):
    train_df = pd.read_csv('../data/asl_data/sign_mnist_train.csv')
    valid_df = pd.read_csv('../data/asl_data/sign_mnist_valid.csv')

    batch_size = 32
    train_data = MyDataset(train_df, device)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    valid_data = MyDataset(valid_df, device)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)
    return train_loader, valid_loader


def init_model(device):
    n_classes = 24
    kernel_size = 3
    flattened_img_size = 75 * 3 * 3

    model = nn.Sequential(
        # First convolution
        nn.Conv2d(IMG_CHS, 25, kernel_size, stride=1, padding=1),  # 25 * 28 * 28
        nn.BatchNorm2d(25),
        nn.ReLU(),
        nn.MaxPool2d(2, stride=2),  # 25 * 14 * 14
        # Second convolution
        nn.Conv2d(25, 50, kernel_size, stride=1, padding=1),  # 50 * 14 * 14
        nn.BatchNorm2d(50),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.MaxPool2d(2, stride=2),  # 50 * 7 * 7
        # Third convolution
        nn.Conv2d(50, 75, kernel_size, stride=1, padding=1),  # 75 * 7 * 7
        nn.BatchNorm2d(75),
        nn.ReLU(),
        nn.MaxPool2d(2, stride=2),  # 75 * 3 * 3
        # Flatten to Dense
        nn.Flatten(),
        nn.Linear(flattened_img_size, 512),
        nn.Dropout(0.3),
        nn.ReLU(),
        nn.Linear(512, n_classes),
    )
    model.to(device)
    if torch.cuda.is_available():
        model.compile()

    return model


def predict(model, arg, expected_output):
    device = next(model.parameters()).device
    if device != arg.device:
        arg = arg.to(device)
    prediction = model(arg)
    indices = prediction.argmax(dim=1)
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


if __name__ == '__main__':
    ASL_NN_MODEL_PATH = '../saved_models/asl_nn_mode_v01.pth'
    device = get_device()
    train_loader, valid_loader = load_data(device)

    import os

    if os.path.exists(ASL_NN_MODEL_PATH):
        model = load_saved_model(ASL_NN_MODEL_PATH)
    else:
        model = init_model(device)
        loss_func = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters())
        train_some_times(model, train_loader, valid_loader, loss_func, optimizer, 20)
        save_model(model, ASL_NN_MODEL_PATH)
    batch = next(iter(valid_loader))
    predict(model, batch[0], batch[1])
