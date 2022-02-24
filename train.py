import datetime
import os
import random
import time
import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.cuda.amp import autocast, GradScaler
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from lib.data import ToDeviceFunction, PrefetchLoader
from lib.utils import print_torch_setup, mkdir, save_on_master, MetricLogger, flatten_dict, SmoothedValue, torchvision
from lib.utils.denormalize import denormalize
from lib.utils.smoothing import gaussian_blur


@hydra.main(config_path='conf', config_name='config')
def main(cfg: DictConfig) -> None:
    if cfg.trainer.print_torch_setup is True:
        print_torch_setup()

    if cfg.trainer.seed is not None:
        random.seed(cfg.trainer.seed)
        torch.manual_seed(cfg.trainer.seed)
        torch.backends.cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    assert torch.cuda.is_available(), 'This code requires a GPU to train'
    torch.backends.cudnn.benchmark = True
    assert cfg.trainer.output_dir, 'You need to specify an output directory'

    mkdir(cfg.trainer.output_dir)
    experiment_name = time.strftime("%Y%m%d-%H%M%S")
    print(f'The current experiment will be tracked as {experiment_name}')
    output_dir = os.path.join(cfg.trainer.output_dir, experiment_name)
    print(f'Results will be saved in {output_dir}')
    writer = SummaryWriter(output_dir)

    # this is just a workaround for now
    # hparams logging to a file and as text into tensorboard
    # it is certainly not perfect... :/
    hparams = flatten_dict(OmegaConf.to_container(cfg, resolve=True))
    hparams_as_str = [str(k) + ' >>> ' + str(v) + '\n' for k, v in hparams.items()]
    # TODO: this seems to not work properly!
    # writer.add_hparams(hparams, metric_dict={'acc': 1}, run_name=experiment_name)
    with open(os.path.join(output_dir, 'hparams.txt'), 'w', encoding='utf-8') as hparams_file:
        for line in hparams_as_str:
            hparams_file.write(line)
    writer.add_text('hparams', '\r\n'.join(hparams_as_str), global_step=0)

    device = torch.device(cfg.trainer.device)
    assert device.type == 'cuda', 'Only GPU based training is supported'

    dataset = instantiate(cfg.dataset.train)

    assert cfg.dataset.val_split is not None, 'Handling a separate validation set is not implemented as of now!'
    train_size = int((1 - cfg.dataset.val_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_sampler_weights = dataset.make_weights_for_dataset_sampling(train_dataset)
    sampler = WeightedRandomSampler(train_sampler_weights, num_samples=cfg.dataset.train_samples_per_epoch,
                                    replacement=True)
    train_collate_fn = dataset.get_collate_fn(mode='train', channels_last=cfg.trainer.channels_last)
    train_dataloader = instantiate(cfg.dataloader.train,
                                   dataset=train_dataset,
                                   collate_fn=train_collate_fn,
                                   sampler=sampler)

    val_collate_fn = dataset.get_collate_fn(mode='val', channels_last=cfg.trainer.channels_last)
    val_dataloader = instantiate(cfg.dataloader.val,
                                 dataset=val_dataset,
                                 collate_fn=val_collate_fn)

    # this handler moves a batch to the GPU as uint8, casts it to a float after transferring it
    # and normalizes the images
    to_device_handler = ToDeviceFunction(device=device, mean=cfg.dataset.mean, std=cfg.dataset.std)

    # the prefetch loader prefetches the next batch onto the GPU which makes up a couple
    # of percent in the training loop
    train_dataloader = PrefetchLoader(loader=train_dataloader,
                                      to_device_handler=to_device_handler)

    # val_dataloader = PrefetchLoader(loader=val_dataloader,
    #                                 to_device_handler=to_device_handler)

    model = instantiate(cfg.models.model,
                        device=device
                        ).to(device)

    if cfg.trainer.channels_last is True:
        model = model.to(memory_format=torch.channels_last)

    if cfg.trainer.anomaly_detection is True:
        torch.autograd.set_detect_anomaly(mode=True)

    params_to_optimize = [
        {"params": [p for p in model.parameters()
                    if p.requires_grad]}
    ]

    optimizer = instantiate(cfg.optimizer, params_to_optimize)

    scaler = GradScaler(enabled=cfg.trainer.amp)

    if cfg.trainer.resume is not None:
        if os.path.isfile(cfg.trainer.resume):
            print("Trying to load checkpoint '{}'".format(cfg.trainer.resume))

            if cfg.trainer.from_u2net_checkpoint is True:
                checkpoint = torch.load(cfg.trainer.resume, map_location=device)
                model.load_state_dict(checkpoint)
            else:
                checkpoint = torch.load(cfg.trainer.resume, map_location=device)
                model.load_state_dict(checkpoint['model'])

                if cfg.trainer.weights_only is False:
                    cfg.trainer.start_epoch = checkpoint['epoch']
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    scaler.load_state_dict(checkpoint['scaler'])

            print(f'Loaded checkpoint {cfg.trainer.resume}. Resuming training at epoch {cfg.trainer.start_epoch}')
        else:
            warnings.warn(f'Checkpoint f{cfg.trainer.resume} not found!')

    print("Start training...")
    start_time = time.time()

    if cfg.trainer.dry_run is True:
        print("Doing dry run, running val on train dataset...")
        # validate_one_epoch(writer, model, train_dataloader, device, 0, cfg.trainer.print_freq)
        return

    for epoch in range(cfg.trainer.start_epoch, cfg.trainer.epochs):
        train_one_epoch(writer, device, model, optimizer, scaler, train_dataloader, epoch, cfg)
        # validate_one_epoch(writer, model, val_dataloader, epoch, cfg)

        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(),
            'epoch': epoch,
            'cfg': cfg}
        save_on_master(
            checkpoint,
            os.path.join(output_dir, 'model_{}.pth'.format(epoch)))
        save_on_master(
            checkpoint,
            os.path.join(output_dir, 'checkpoint.pth'))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def create_metric_logger(train, epoch, writer):
    if train:
        prefix = 'train'
    else:
        prefix = 'val'

    metric_logger = MetricLogger(epoch=epoch, delimiter="  ", writer=writer, experiment_prefix=prefix)

    if train:
        metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value}'), log=False)
        metric_logger.add_meter('samples/s', SmoothedValue(window_size=10, fmt='{value}'), log=True, log_value='median',
                                title='samples per second')
        metric_logger.add_meter('loss', SmoothedValue(), log=True, log_value='global_avg',
                                title='loss')

    return metric_logger


def criterion(aux, y, metadata, device):
    # aux ^= [d0, d1, d2, d3, d4, d5, d6]

    def masked_l1_loss(y_hat, y, mask):
        loss = F.l1_loss(y_hat, y, reduction='none')
        loss = (loss * mask.float()).sum()
        non_zero_elements = mask.sum()
        return loss / non_zero_elements

    mask = y[:, 0]
    smoothed_mask = gaussian_blur(
        mask.unsqueeze(dim=1), (9, 9), (2.5, 2.5)).squeeze(dim=1)
    unknown_mask = y[:, 1]

    l1_mask = torch.ones(mask.shape, device=device)
    l1_details_mask = torch.zeros(mask.shape, device=device)

    # i synthesised some detailed masks using pymatting.github.io
    # by synthesising trimaps from segmentation masks and use these
    # in an additional loss to let the model learn the unknown areas
    # between foreground and background. this is not perfect as the generated
    # trimaps and masks are not super accurate, but it seems to go in the right
    # direction.
    detailed_masks = [x['detailed_masks'] for x in metadata]
    for idx, detailed_mask in enumerate(detailed_masks):
        if not detailed_mask:
            l1_mask[idx] = l1_mask[idx] - unknown_mask[idx]
        else:
            l1_details_mask[idx] = unknown_mask[idx]

    loss = 0
    for output in aux:
        loss += 2 * masked_l1_loss(output, mask, l1_mask)
        # this loss should give some learning signals to focus on unknown areas
        loss += 3 * masked_l1_loss(output, mask, l1_details_mask)
        # i'm not quite sure if this loss gives the right incentive, the idea
        # is to blur the segmentation mask a bit to reduce background bleeding
        # caused by bad labels, preliminary results seem to be quite ok.
        loss += F.mse_loss(output, smoothed_mask)

    aux = {
        'l1_mask': l1_mask,
        'l1_detailed_mask': l1_details_mask,
        'mask': mask,
        'smoothed_mask': smoothed_mask
    }

    return loss, aux


def train_one_epoch(writer, device, model, optimizer, scaler, data_loader, epoch, cfg):
    model.train()

    metric_logger = create_metric_logger(train=True, epoch=epoch, writer=writer)

    for x, y, metadata in metric_logger.log_every(data_loader, cfg.trainer.print_freq):
        start_time = time.time()

        with autocast(enabled=cfg.trainer.amp):
            y_hat, aux_outputs = model(x)
            loss, aux = criterion(aux_outputs, y, metadata, device)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        metric_logger.update(
            loss=loss.item(),
            lr=optimizer.param_groups[0]["lr"])

        metric_logger.meters['samples/s'].update(x.size(0) / (time.time() - start_time))

        if random.random() < .1:
            sample = denormalize(x[:4], mean=cfg.dataset.mean, std=cfg.dataset.std)
            sample_foreground = y_hat[:4].unsqueeze(dim=1).repeat(1,3,1, 1) * sample

            writer.add_image(
                f'train-metrics/sample',
                torchvision.utils.make_grid(
                    [torchvision.utils.make_grid(sample, nrow=4),
                     torchvision.utils.make_grid(sample_foreground),
                     torchvision.utils.make_grid(y_hat[:4].unsqueeze(dim=1), nrow=4)], nrow=1),
                metric_logger.global_step)

            writer.add_image(
                f'train-metrics/loss insights',
                torchvision.utils.make_grid(
                    [torchvision.utils.make_grid(aux['l1_mask'][:4].unsqueeze(dim=1), nrow=4),
                     torchvision.utils.make_grid(aux['l1_detailed_mask'][:4].unsqueeze(dim=1), nrow=4),
                     torchvision.utils.make_grid(aux['smoothed_mask'][:4].unsqueeze(dim=1), nrow=4),
                     torchvision.utils.make_grid(aux['mask'][:4].unsqueeze(dim=1), nrow=4)], nrow=1),
                metric_logger.global_step)


if __name__ == "__main__":
    main()
