import json

import torch
import torchvision.io as tv_io
from torchvision.models import VGG16_Weights
from torchvision.models import vgg16

from utils import get_device, show_image


def load_model_and_pretransforms(device):
    weights = VGG16_Weights.DEFAULT
    model = vgg16(weights=weights)
    model.to(device)

    pre_trans = weights.transforms()
    # In MAC, `antialias` is not implemented in MPS
    pre_trans.antialias = False

    return model, pre_trans


def load_vgg_classes():
    vgg_classes = json.load(open('./data/imagenet_class_index.json'))
    return vgg_classes


def load_and_process_image(file_path, pre_transforms, device):
    image = tv_io.read_image(file_path).to(device)
    print(f'Original image shape: {image.shape}')
    image = pre_transforms(image)
    print(f'After transforming, image shape: {image.shape}')
    # turn into a batch
    image = image.unsqueeze(0)
    return image


def predict(model, pre_transforms, image_path):
    show_image(image_path)
    dv = next(model.parameters()).device
    image = load_and_process_image(image_path, pre_transforms, dv)
    output = model(image)
    return output


def readable_prediction(model, pre_transforms, image_path):
    vgg_classes = load_vgg_classes()
    output = predict(model, pre_transforms, image_path)
    output = output[0]  # un-batch
    predictions = torch.topk(output, 3)
    indices = predictions.indices.tolist()
    pred_classes = [(idx, vgg_classes[str(idx)][1]) for idx in indices]
    print(f'Top 3 results: {pred_classes}\n')

    return predictions


def doggy_door(model, pre_transforms, image_path):
    output = predict(model, pre_transforms, image_path)
    idx = output.argmax(dim=1).item()
    print(f'Predicted index: {idx}')
    if 151 <= idx <= 268:
        print('Doggy come on in!')
    elif 281 <= idx <= 285:
        print('Kitty stay inside!')
    else:
        print('You are not a dog! Stay outside!')
    print()


if __name__ == '__main__':
    dvc = get_device()
    mdl, pre_trans = load_model_and_pretransforms(dvc)
    readable_prediction(mdl, pre_trans, './data/doggy_door_images/happy_dog.jpg')
    readable_prediction(mdl, pre_trans, './data/doggy_door_images/brown_bear.jpg')
    readable_prediction(mdl, pre_trans, './data/doggy_door_images/sleepy_cat.jpg')

    doggy_door(mdl, pre_trans, './data/doggy_door_images/happy_dog.jpg')
    doggy_door(mdl, pre_trans, './data/doggy_door_images/brown_bear.jpg')
    doggy_door(mdl, pre_trans, './data/doggy_door_images/sleepy_cat.jpg')
