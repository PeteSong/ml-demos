import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as F


def get_device():
    s = 'cpu'
    if torch.cuda.is_available():
        s = 'cuda'
    elif torch.backends.mps.is_available():
        s = 'mps'
    # print(s)
    return torch.device(s)


def get_batch_accuracy(output, y, N):
    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(y.view_as(pred)).sum().item()
    return correct / N


def save_state_model(model, fp):
    torch.save(model.state_dict(), fp)


def save_model(model, fp):
    torch.save(model, fp)


def load_saved_state_model(model, fp):
    model.load_state_dict(torch.load(fp))
    model.to(get_device())
    model.eval()

    return model


def load_saved_model(fp):
    device = get_device()
    model = torch.load(fp, map_location=device, weights_only=False)
    model.eval()
    return model


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
        output = model(x)
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
            output = model(x)

            loss += loss_func(output, y).item()
            accuracy += get_batch_accuracy(output, y, valid_N)
    print(f'Valid => Loss:{loss:.4f}, Accuracy:{accuracy:.4f}')


def train_some_times(model, train_loader, valid_loader, loss_func, optimizer, epochs=5):
    for epoch in range(epochs):
        print('Epoch: {}'.format(epoch))
        train(model, train_loader, loss_func, optimizer)
        validate(model, valid_loader, loss_func)


def show_image(image_path):
    image = mpimg.imread(image_path)
    plt.imshow(image, cmap='gray')
    # plt.ion()
    plt.pause(0.5)
    # plt.show()


def show_processed_image(processed_image):
    plot_image = F.to_pil_image(torch.squeeze(processed_image))
    plt.imshow(plot_image, cmap='gray')
    plt.pause(0.3)
