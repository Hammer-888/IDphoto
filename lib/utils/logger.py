import datetime
import time
from collections import defaultdict, deque

import psutil
import torch
import torch.distributed as dist

from .distributed import is_dist_avail_and_initialized


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, epoch, delimiter="\t", writer=None, experiment_prefix=None):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.epoch = epoch
        self.writer = writer
        self.experiment_prefix = experiment_prefix
        self.logged_meters = {}
        self.global_step = 0

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter, log, log_value='value', title=None, prefix=None):
        self.meters[name] = meter

        if log:
            self.logged_meters[name] = {
                'value': log_value,
                'title': title,
                'prefix': prefix
            }

    def log_meters(self):
        for k, v in self.logged_meters.items():
            value = self.meters[k]
            title = v['title'] if v['title'] is not None else k
            prefix = v['prefix'] if v['prefix'] is not None else self.experiment_prefix

            scalar = 0
            if v['value'] == 'value':
                scalar = value.value
            elif v['value'] == 'median':
                scalar = value.median
            elif v['value'] == 'max':
                scalar = value.max
            elif v['value'] == 'global_avg':
                scalar = value.global_avg
            elif v['value'] == 'avg':
                scalar = value.avg
            elif v['value'] == 'total':
                scalar = value.total

            self.writer.add_scalar(f'{prefix}/{title}',
                                   scalar,
                                   self.global_step)

    def log_meters_epoch(self):
        for k, v in self.logged_meters.items():
            value = self.meters[k]
            title = v['title'] if v['title'] is not None else k
            prefix = v['prefix'] if v['prefix'] is not None else self.experiment_prefix

            self.writer.add_scalar(f'{prefix}/epoch/{title}',
                                   value.global_avg,
                                   self.epoch)

    def log_every(self, iterable, print_freq):
        i = 0
        header = 'Epoch: [{}]'.format(self.epoch)
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0

        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)

            self.global_step = self.epoch * len(iterable) + i

            if i % print_freq == 0:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

                if self.writer is not None:
                    self.writer.add_scalar(f'{self.experiment_prefix}/seconds per step',
                                           iter_time.median,
                                           self.global_step)

                    self.writer.add_scalar(f'{self.experiment_prefix}/seconds to yield batch',
                                           data_time.median,
                                           self.global_step)

                    self.log_meters()

                    if torch.cuda.is_available():
                        self.writer.add_scalar(f'{self.experiment_prefix}/max memory',
                                               torch.cuda.max_memory_allocated() / MB,
                                               self.global_step)

                    self.writer.add_scalar(f'{self.experiment_prefix}/cpu percent',
                                           psutil.cpu_percent(),
                                           self.global_step)

                    self.writer.add_scalar(f'{self.experiment_prefix}/virtual memory %',
                                           psutil.virtual_memory().percent,
                                           self.global_step)

                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_delta = datetime.timedelta(seconds=int(total_time))

        if self.writer is not None:
            self.writer.add_scalar(f'{self.experiment_prefix}/seconds per epoch',
                                   total_time_delta.total_seconds(),
                                   self.epoch * len(iterable) + i)
        self.log_meters_epoch()
        print('{} Total time: {}'.format(header, str(total_time_delta)))