import torch
from torch.utils.data import DataLoader


class PrefetchLoader:
    def __init__(self, loader, to_device_handler):
        self.loader = loader
        self.to_device_handler = to_device_handler

    def __iter__(self):
        stream = torch.cuda.Stream()
        first = True

        for batch in self.loader:
            with torch.cuda.stream(stream):
                next_input, next_target, next_metadata = self.to_device_handler(batch)

            if not first:
                yield input, target, metadata
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)

            input = next_input
            target = next_target
            metadata = next_metadata

        yield input, target, metadata

    def __len__(self):
        return len(self.loader)

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def dataset(self):
        return self.loader.dataset


class InfiniteDataLoader(DataLoader):
    """ Dataloader that reuses workers
    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)
