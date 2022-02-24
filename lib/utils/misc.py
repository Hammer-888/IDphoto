import errno
import os
import sys
from argparse import Namespace
from collections import MutableMapping
from subprocess import run, PIPE
from typing import Dict, Any

import torch
import torchvision
from torch.autograd import Variable


def print_torch_setup() -> None:
    """
    Print PyTorch and GPU libraries/availability, then check that tensors can be generated.
    https://github.com/benjaminlackey/bentorched/blob/main/utils.py
    """
    print("Python:\t\t\t", sys.version)
    print("PyTorch：\t\t", torch.__version__)
    print("Torchvision:\t\t", torchvision.__version__)
    print("CUDA:\t\t\t", torch.version.cuda)
    print("cuDNN:\t\t\t", torch.backends.cudnn.version())
    print("Arch:\t\t\t", torch._C._cuda_getArchFlags())
    print("CUDA is available:\t", torch.cuda.is_available())
    print("Number of CUDA devices:\t", torch.cuda.device_count())
    print("Current CUDA device:\t", torch.cuda.current_device())
    print("Random tensor on CPU:\t", torch.rand(5))
    print("Random tensor on GPU:\t", torch.rand(5).cuda())

    print("\nnvidia-smi (The CUDA Version listed is the maximum supported by the driver, not the actual version.):")
    result = run("nvidia-smi", stdout=PIPE)
    smi = result.stdout.decode("utf-8")
    print('\n'.join(smi.split('\n')[:12]))


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def to_one_hot(y, n_dims=None):
    """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
    y_tensor = y.data if isinstance(y, Variable) else y
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return Variable(y_one_hot) if isinstance(y, Variable) else y_one_hot


# taken from pytorch rnn utils but changed to allow a fixed max_len
def pad_sequence(sequences, max_len, batch_first=False, padding_value=0.0):
    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]

    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = sequences[0].new_full(out_dims, padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_tensor[i, :length, ...] = tensor
        else:
            out_tensor[:length, i, ...] = tensor

    return out_tensor


# https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pytorch_lightning/loggers/base.py#L45
def flatten_dict(params: Dict[Any, Any], delimiter: str = '.') -> Dict[str, Any]:
    def _dict_generator(input_dict, prefixes=None):
        prefixes = prefixes[:] if prefixes else []
        if isinstance(input_dict, MutableMapping):
            for key, value in input_dict.items():
                key = str(key)
                if isinstance(value, (MutableMapping, Namespace)):
                    value = vars(value) if isinstance(value, Namespace) else value
                    for d in _dict_generator(value, prefixes + [key]):
                        yield d
                else:
                    if isinstance(value, (int, float, str, bool, torch.Tensor)) is False:
                        value = str(value)

                    yield prefixes + [key, value if value is not None else str(None)]
        else:
            yield prefixes + [input_dict if input_dict is None else str(input_dict)]

    return {delimiter.join(keys): val for *keys, val in _dict_generator(params)}


def set_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad = requires_grad