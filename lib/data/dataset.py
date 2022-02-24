import glob
import os
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image
from hydra.utils import instantiate
from torch.utils.data import Dataset, Subset

from .collate_function import CollateFunction
from .transforms import Compose


class SegmentationDataset(Dataset):
    def __init__(self, configurations):
        self.configurations = configurations

        self.images = []
        self.masks = []
        self.metadata = []

        for configuration in self.configurations:
            images = glob.glob(os.path.join(configuration.images_path, '*.png'))

            masks = []
            metadata = []
            for image in images:
                mask_path = os.path.join(configuration.masks_path, f'{os.path.basename(image)}')
                masks.append(mask_path)
                metadata.append({
                    'image_path': image,
                    'mask_path': mask_path,
                    'dataset_identifier': configuration.identifier,
                    'detailed_masks': configuration.detailed_masks,
                    'rgba_masks': configuration.rgba_masks
                })

            self.images.extend(images)
            self.masks.extend(masks)
            self.metadata.extend(metadata)

    def __getitem__(self, index):
        img_path = self.images[index]
        target_path = self.masks[index]
        metadata = self.metadata[index]

        img = Image.open(img_path).convert('RGB')

        if metadata['rgba_masks'] is True:
            target = Image.open(target_path).convert('RGBA')
            target = np.asarray(target)
            target = target[:, :, 3]
            target = Image.fromarray(target).convert('L')
        else:
            target = Image.open(target_path)

        background, unknown, foreground = self.generate_trimap(np.asarray(target))
        target = Image.merge('RGB', (target,
                                      Image.fromarray(unknown),
                                      Image.fromarray(foreground)))

        return img, target, metadata

    def __len__(self):
        return len(self.images)

    def get_collate_fn(self, mode, channels_last):
        assert mode in ['train', 'val'], 'mode must be either train or val!'

        transforms = {}
        for configuration in self.configurations:
            steps = configuration.transforms.train if mode == 'train' else configuration.transforms.val
            transforms[configuration.identifier] = [instantiate(step) for step in steps]

        return CollateFunction(Compose(transforms, metadata_key='dataset_identifier'), channels_last)

    def generate_trimap(self, mask):
        dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        dilated = cv2.dilate(mask, dilation_kernel, iterations=5)

        erosion_kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(mask, erosion_kernel, iterations=3)

        background = np.zeros(mask.shape, dtype=np.uint8)
        background[dilated < 128] = 255

        unknown = np.zeros(mask.shape, dtype=np.uint8)
        unknown.fill(255)
        unknown[eroded > 128] = 0
        unknown[dilated < 128] = 0

        foreground = np.zeros(mask.shape, dtype=np.uint8)
        foreground[eroded > 128] = 255

        return background, unknown, foreground

    def make_weights_for_dataset_sampling(self, dataset):
        if isinstance(dataset, Subset):
            indices = range(len(dataset.indices))
        elif isinstance(dataset, SegmentationDataset):
            indices = range(len(dataset))

        counts = defaultdict(int)
        for item in self.metadata:
            counts[item['dataset_identifier']] += 1

        weights_per_class = {x.identifier: x.weight for x in self.configurations}

        for key, value in counts.items():
            weights_per_class[key] = len(self.metadata) / float(value) * weights_per_class[key]

        weights = [0] * len(indices)
        for idx in indices:
            if isinstance(dataset, Subset):
                item = self.metadata[dataset.indices[idx]]
            else:
                item = self.metadata[idx]

            weights[idx] = weights_per_class[item['dataset_identifier']]

        return weights
