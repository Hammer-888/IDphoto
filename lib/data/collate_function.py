import numpy as np
import torch


class CollateFunction:
    def __init__(self, transforms, channels_last):
        self.transforms = transforms
        self.channels_last = channels_last

    def __call__(self, batch):
        tensor, target_tensor, metadata = None, None, []

        for i, sample in enumerate(batch):
            if self.transforms is not None:
                sample = self.transforms(*sample)

            if tensor is None:
                h, w = sample[0].size
                memory_format = torch.channels_last if self.channels_last else torch.contiguous_format

                tensor = torch.zeros((len(batch), 3, h, w), dtype=torch.uint8).contiguous(
                    memory_format=memory_format)

                # channels_last is not necessary here
                # note, the targets contain 3 channels:
                #   - semantic maps, foreground (from trimap), unknown (from trimap)
                target_tensor = torch.zeros((len(batch), 3, h, w), dtype=torch.uint8).contiguous(
                    memory_format=torch.contiguous_format)

            # this should not be np.array but np.isarray, however there is a bug in PyTorch
            # because the image is not writeable
            # this bug causes millions of warnings, this is why we're sticking to np.array(..., copy=True)
            # which is however less efficient

            x, y, sample_metadata = sample
            x = np.array(x).transpose(2, 0, 1)  # C x H x W
            y = np.array(y).transpose(2, 0, 1)  # C x H x W

            tensor[i] += torch.from_numpy(x)
            target_tensor[i] += torch.from_numpy(y)
            metadata.append(sample_metadata)

        return tensor, target_tensor, metadata
