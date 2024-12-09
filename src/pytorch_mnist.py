# HelloWorld in the Deep Learning: MNIST
# device: MBP M3

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.v2 as transforms
from torch.optim import Adam
from torch.utils.data import DataLoader

from utils import get_device, save_model, load_saved_model, train_some_times


# import torchvision.transforms.functional as F
# import matplotlib.pyplot as plt

# print(f'torch version: {torch.__version__}, torchvision version: {torchvision.__version__}')


########## Load data

def load_data():
    def check_dataset():
        train_N = len(train_set)
        valid_N = len(valid_set)

        print(f'train_set => type:{type(train_set)}, size:{train_N}, value:{train_set}')
        print(f'valid_set => type:{type(valid_set)}, size:{valid_N}, value:{valid_set}')

        x_0, y_0 = train_set[0]
        print(f'x_0 => type:{type(x_0)}, value:{x_0}')
        print(f'y_0 => type:{type(y_0)}, value:{y_0}')

        # x_0_tensor = trans(x_0)
        # print(f'x_0_tensor => type:{type(x_0_tensor)}, size:{x_0_tensor.size()}, device:{x_0_tensor.device}, value:{x_0_tensor}')
        # 把这个 张量 交给 GPU 处理
        # x_0_tensor.to(device)
        # print(f'x_0_tensor -> device:{x_0_tensor.to(device).device}')

        # image = F.to_pil_image(x_0_tensor)
        # plt.imshow(image, cmap='gray')

    train_set = torchvision.datasets.MNIST('./data/', train=True, download=True)
    valid_set = torchvision.datasets.MNIST('./data/', train=False, download=True)

    # check_dataset()

    # 一个 transform ，将 PIL 转换成 Tensor(张量)
    trans = transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)])
    train_set.transform = trans
    valid_set.transform = trans

    # check_dataset()

    batch_size = 32
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size)

    return train_set, valid_set, train_loader, valid_loader


########## Define model

def init_model():
    # 通道数 * 垂直像素数 * 水平像素数
    # MNIST 数据集包含 70,000 个手写体数字的灰度图像，大小是 28*28
    #   通道数：这些图像是黑白的，所以只有 1 个颜色通道。
    #   垂直像素数/水平像素数：28
    input_size = 1 * 28 * 28

    # 由于这个任务是猜测图像属于 10 个可能类别中的哪一个，因此将有 10 个输出.
    n_classes = 10

    # 神经网络 层级 定义
    layers = [
        nn.Flatten(),  # 将 n 维数据转换成 向量
        nn.Linear(input_size, 512),  # Input
        nn.ReLU(),  # Activation for input
        nn.Linear(512, 512),  # Hidden
        nn.ReLU(),  # Activation for hidden
        nn.Linear(512, n_classes),  # Output
    ]

    model = nn.Sequential(*layers)
    model.to(get_device())

    ## in mac, after compile, the inductor does not support MPS
    if torch.cuda.is_available():
        model = torch.compile(model)

    return model


def predict(model, arg, expected_output):
    device = next(model.parameters()).device
    if device != arg.device:
        arg = arg.to(device)
    prediction = model(arg)
    indices = prediction.argmax(dim=1, keepdim=True)
    if isinstance(expected_output, torch.Tensor):
        isPassed = indices.eq(expected_output)
    else:
        isPassed = indices.item() == expected_output
    print(f'''
    prediction: {prediction}
    Indices of the maximum values in Prediction: {indices}
    expected output: {expected_output}
    Is passed: {isPassed}
    ''')


########## Train model


if __name__ == '__main__':
    MNIST_MODEL_PATH = './saved_models/mnist_mode.pth'
    train_set, valid_set, train_loader, valid_loader = load_data()

    import os

    if os.path.exists(MNIST_MODEL_PATH):
        model = load_saved_model(MNIST_MODEL_PATH)
    else:
        model = init_model()
        loss_func = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters())
        train_some_times(model, train_loader, valid_loader, loss_func, optimizer)
        save_model(model, MNIST_MODEL_PATH)
    x_0, y_0 = valid_set[0]
    predict(model, x_0, y_0)
