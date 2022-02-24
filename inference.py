import os
from collections import defaultdict
from glob import glob

import PIL
import numpy as np

import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, CenterCrop
import torchvision.transforms.functional as F
from pymatting import *
from lib import U2NET_full
from lib.utils.oom import free_up_memory
import cv2


# def load_samples(folder_path='./dataset/demo'):
#     assert os.path.isdir(folder_path), f'Unable to open {folder_path}'
#     samples = glob(os.path.join(folder_path, f'*.jpeg'))
#     return samples


# device = 'cuda'
# samples = load_samples()
# model_select, sample_select = create_ui(samples)


def square_pad(image, fill=255):
    w, h = image.size
    max_wh = np.max([w, h])
    hp = int((max_wh - w) / 2)
    vp = int((max_wh - h) / 2)
    padding = (hp, vp, hp, vp)
    return F.pad(image, padding, fill, "constant")


def get_transform():
    transforms = []
    # transforms.append(Resize(440))  # TBD: keep aspect ratio
    transforms.append(ToTensor())
    transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))

    return Compose(transforms)


model_select = "u2net_human_seg.pth"
device = "cpu"
checkpoint = torch.load(f"./checkpoints/{model_select}", map_location=device)
model = U2NET_full().to(device=device)

if "model" in checkpoint:
    model.load_state_dict(checkpoint["model"])
else:
    model.load_state_dict(checkpoint)


# image = Image.open(sample_select).convert('RGB')
def run(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    # image = square_pad(image, 0)
    image = image.resize((295, 413), Image.ANTIALIAS)

    transforms = get_transform()

    model.eval()
    with torch.no_grad():
        x = transforms(image)
        x = x.to(device=device).unsqueeze(dim=0)
        y_hat, _ = model(x)

        alpha_image = y_hat.mul(255)
        alpha_image = Image.fromarray(
            alpha_image.squeeze().cpu().detach().numpy()
        ).convert("L")

    image = np.asarray(image)
    background = np.zeros(image.shape)
    background[:, :] = [255 / 255, 255 / 255, 255 / 255]

    alpha = y_hat.squeeze().cpu().detach()
    alpha = np.asarray(alpha)
    # alpha = (alpha * 255).astype(np.uint8)
    image = image.astype(np.float32) / 255

    foreground = estimate_foreground_ml(
        image, alpha
    )  # , n_big_iterations=1, n_small_iterations=1, regularization=10e-10

    new_image = blend(foreground, background, alpha)
    new_image = (new_image * 255).astype(np.uint8)
    new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    # del y_hat
    return new_image


# free_up_memory()
