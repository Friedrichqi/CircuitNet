# Copyright 2022 CircuitNet. All rights reserved.

import os
import json
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from torch.utils.data.distributed import DistributedSampler

from datasets.build_dataset import build_dataset
from utils.losses import build_loss
from models.build_model import build_model
from utils.configs import Parser
from math import cos, pi
import sys, os, subprocess
import argparse

def checkpoint(model, epoch, save_path, rank):
    """Save checkpoint only on master process (rank 0)"""
    if rank == 0:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        model_out_path = f"./{save_path}/model_iters_{epoch}.pth"
        # Save model.module's state_dict if it's wrapped by DDP
        state_dict = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
        torch.save({'state_dict': state_dict}, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

class CosineRestartLr(object):
    def __init__(self,
                 base_lr,
                 periods,
                 restart_weights = [1],
                 min_lr = None,
                 min_lr_ratio = None):
        self.periods = periods
        self.min_lr = min_lr
        self.min_lr_ratio = min_lr_ratio
        self.restart_weights = restart_weights
        super().__init__()

        self.cumulative_periods = [
            sum(self.periods[0:i + 1]) for i in range(0, len(self.periods))
        ]

        self.base_lr = base_lr

    def annealing_cos(self, start: float,
                    end: float,
                    factor: float,
                    weight: float = 1.) -> float:
        cos_out = cos(pi * factor) + 1
        return end + 0.5 * weight * (start - end) * cos_out

    def get_position_from_periods(self, iteration: int, cumulative_periods):
        for i, period in enumerate(cumulative_periods):
            if iteration < period:
                return i
        raise ValueError(f'Current iteration {iteration} exceeds '
                        f'cumulative_periods {cumulative_periods}')


    def get_lr(self, iter_num, base_lr: float):
        target_lr = self.min_lr  # type:ignore

        idx = self.get_position_from_periods(iter_num, self.cumulative_periods)
        current_weight = self.restart_weights[idx]
        nearest_restart = 0 if idx == 0 else self.cumulative_periods[idx - 1]
        current_periods = self.periods[idx]

        alpha = min((iter_num - nearest_restart) / current_periods, 1)
        return self.annealing_cos(base_lr, target_lr, alpha, current_weight)

    
    def _set_lr(self, optimizer, lr_groups):
        if isinstance(optimizer, dict):
            for k, optim in optimizer.items():
                for param_group, lr in zip(optim.param_groups, lr_groups[k]):
                    param_group['lr'] = lr
        else:
            for param_group, lr in zip(optimizer.param_groups,
                                        lr_groups):
                param_group['lr'] = lr

    def get_regular_lr(self, iter_num):
        return [self.get_lr(iter_num, _base_lr) for _base_lr in self.base_lr]  # iters

    def set_init_lr(self, optimizer):
        for group in optimizer.param_groups:  # type: ignore
            group.setdefault('initial_lr', group['lr'])
            self.base_lr = [group['initial_lr'] for group in optimizer.param_groups  # type: ignore
        ]


def train():
    # Initialize argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0)
    args, unknown = parser.parse_known_args()
    local_rank = args.local_rank

    # Initialize distributed process group
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # Parse arguments
    argp = Parser()
    arg = argp.parser.parse_args()
    arg_dict = vars(arg)
    if arg.arg_file is not None:
        with open(arg.arg_file, 'rt') as f:
            arg_dict.update(json.load(f))

    # Only master process handles logging and saving
    if rank == 0:
        if not os.path.exists(arg_dict['save_path']):
            os.makedirs(arg_dict['save_path'])
        with open(os.path.join(arg_dict['save_path'], 'arg.json'), 'wt') as f:
            json.dump(arg_dict, f, indent=4)

    arg_dict['ann_file'] = arg_dict['ann_file_train']
    arg_dict['test_mode'] = False

    # Build dataset and dataloader with DistributedSampler
    if rank == 0:
        print('===> Loading datasets')
    train_dataset = build_dataset(arg_dict)
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=arg_dict['batch_size'],
        sampler=train_sampler,
        num_workers=16,
        pin_memory=True,
        drop_last=True
    )

    # Build model and wrap with DDP
    if rank == 0:
        print('===> Building model')
    model = build_model(arg_dict).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # Build loss and optimizer
    loss = build_loss(arg_dict)
    optimizer = optim.AdamW(model.parameters(), lr=arg_dict['lr'], betas=(0.9, 0.999), weight_decay=arg_dict['weight_decay'])

    # Learning rate scheduler
    cosine_lr = CosineRestartLr(arg_dict['lr'], [arg_dict['max_iters']], [1], 1e-7)
    cosine_lr.set_init_lr(optimizer)

    # Training variables
    epoch_loss = 0
    iter_num = 0
    print_freq = 100
    save_freq = 10000

    # Main training loop
    while iter_num < arg_dict['max_iters']:
        # Set epoch for sampler to ensure proper shuffling
        train_sampler.set_epoch(iter_num)
        
        # Progress bar only on master process
        if rank == 0:
            bar = tqdm(total=print_freq, desc=f'Iteration {iter_num}')
        else:
            bar = None

        for i, (feature, label, _) in enumerate(dataloader):
            # Stop after print_freq batches
            if i >= print_freq:
                break
                
            # Move data to device
            input = feature.to(device, non_blocking=True)
            target = label.to(device, non_blocking=True)

            # Adjust learning rate
            regular_lr = cosine_lr.get_regular_lr(iter_num)
            cosine_lr._set_lr(optimizer, regular_lr)

            # Forward pass
            prediction = model(input)
            pixel_loss = loss(prediction, target)

            # Backward pass
            optimizer.zero_grad()
            pixel_loss.backward()
            optimizer.step()

            # Update metrics
            epoch_loss += pixel_loss.item()
            iter_num += 1

            # Update progress bar on master
            if bar is not None:
                bar.update(1)

            # Exit loop if max iterations reached
            if iter_num >= arg_dict['max_iters']:
                break

        # Close progress bar
        if bar is not None:
            bar.close()

        # Synchronize and average loss across all processes
        dist_loss = torch.tensor(epoch_loss, device=device)
        dist.all_reduce(dist_loss, op=dist.ReduceOp.SUM)
        avg_loss = dist_loss.item() / (print_freq * world_size)

        # Print loss only on master
        if rank == 0:
            print("===> Iters[{}]({}/{}): Loss: {:.4f}".format(
                iter_num, iter_num, arg_dict['max_iters'], avg_loss))

        # Save checkpoint on master
        if iter_num % save_freq == 0:
            checkpoint(model, iter_num, arg_dict['save_path'], rank)

        epoch_loss = 0

    # Cleanup distributed process group
    dist.destroy_process_group()

if __name__ == "__main__":
    train()
