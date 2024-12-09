import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from utils import get_device, load_saved_model, save_model, train_some_times


class MyDataset(Dataset):
    def __init__(self, x_df, y_df, device):
        self.xs = torch.tensor(x_df).float().to(device)
        self.ys = torch.tensor(y_df).to(device)

    def __getitem__(self, idx):
        x = self.xs[idx]
        y = self.ys[idx]
        return x, y

    def __len__(self):
        return len(self.xs)


def load_data(device):
    def check_data():
        print(
            f'''
        train x => shape: {x_train.shape}
        train y => shape: {y_train.shape}

        valid x => shape: {x_valid.shape}
        valid y => shape: {y_valid.shape}
        '''
        )

    def imshow_sample_data():
        plt.figure(figsize=(40, 40))
        num_images = 20
        for i in range(num_images):
            row = x_train[i]
            label = y_train[i]
            image = row.reshape(28, 28)
            plt.subplot(1, num_images, i + 1)
            plt.title(label, fontdict={'fontsize': 30})
            plt.axis('off')
            plt.imshow(image, cmap='gray')

    train_df = pd.read_csv('../data/asl_data/sign_mnist_train.csv')
    valid_df = pd.read_csv('../data/asl_data/sign_mnist_valid.csv')

    y_train = train_df.pop('label')
    x_train = train_df.to_numpy()

    y_valid = valid_df.pop('label')
    x_valid = valid_df.to_numpy()

    check_data()
    imshow_sample_data()

    MAX_PIXEL_VALUE = 255
    # scaling the data (standardization)
    x_train = x_train / MAX_PIXEL_VALUE
    x_valid = x_valid / MAX_PIXEL_VALUE

    BATCH_SIZE = 32
    train_data = MyDataset(x_train, y_train, device)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    # train_N = len(train_loader.dataset)

    valid_data = MyDataset(x_valid, y_valid, device)
    valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=True)
    # valid_N = len(valid_loader.dataset)
    return train_loader, valid_loader


def init_model(device):
    input_size = 1 * 28 * 28
    n_class = 24
    FEATURE_SIZE = 512
    model = nn.Sequential(
        nn.Flatten(),  # 将 n维张量 变成 二维向量
        nn.Linear(input_size, FEATURE_SIZE),  # input layer
        nn.ReLU(),  # Activation for input
        nn.Linear(FEATURE_SIZE, FEATURE_SIZE),  # hidden layer
        nn.ReLU(),  # Activation for hidden layer
        nn.Linear(FEATURE_SIZE, n_class),  # output layer
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
    if isinstance(expected_output, torch.Tensor):
        is_passed = indices.eq(expected_output)
    else:
        is_passed = indices.item() == expected_output
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
    ASL_MODEL_PATH = '../saved_models/asl_mode.pth'
    device = get_device()
    train_loader, valid_loader = load_data(device)

    import os

    if os.path.exists(ASL_MODEL_PATH):
        model = load_saved_model(ASL_MODEL_PATH)
    else:
        model = init_model(device)
        loss_func = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters())
        train_some_times(model, train_loader, valid_loader, loss_func, optimizer, 20)
        save_model(model, ASL_MODEL_PATH)
    batch = next(iter(valid_loader))
    predict(model, batch[0], batch[1])
