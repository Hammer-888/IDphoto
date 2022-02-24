import torch


class ToDeviceFunction:
    def __init__(self, device, mean, std):
        self.device = device
        self.mean = torch.tensor([x * 255 for x in mean]).to(device).view(1, 3, 1, 1)
        self.std = torch.tensor([x * 255 for x in std]).to(device).view(1, 3, 1, 1)

    def __call__(self, sample):
        x, y, metadata = sample

        x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
        x, y = x.float(), y.float()
        x.sub_(self.mean.squeeze(dim=0)).div_(self.std.squeeze(dim=0))
        y.div_(255)

        return x, y, metadata
