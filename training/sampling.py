"""Sample decision rows with explicit episode/task probabilities."""

from collections import Counter, defaultdict
import json
import os

import torch
from torch.utils.data import Sampler


class EpisodeSampler(Sampler):
    """One global deterministic stream; Accelerate shards it across ranks."""

    def __init__(self, dataset, *, strategy, seed, num_samples=None, max_repeats=None):
        if strategy not in {'episode_uniform', 'task_episode_uniform'}:
            raise ValueError(f'Unknown sampling strategy: {strategy}')
        index = dataset.with_format(None)
        self.tasks = list(index['task_id'])
        self.episodes = list(zip(self.tasks, index['trajectory_id']))
        if any(not task or not episode for task, episode in self.episodes):
            raise ValueError('Episode sampling requires task_id and trajectory_id for every selected row')
        counts = Counter(self.episodes)
        task_episodes = defaultdict(set)
        for task, episode in self.episodes:
            task_episodes[task].add(episode)
        self.weights = torch.tensor([
            1.0 / counts[episode] / (len(task_episodes[episode[0]]) if strategy == 'task_episode_uniform' else 1)
            for episode in self.episodes
        ], dtype=torch.double)
        world_size = int(os.environ.get('WORLD_SIZE', '1'))
        self.num_samples = len(dataset) // world_size * world_size if num_samples is None else num_samples
        if self.num_samples % world_size:
            raise ValueError('samples_per_epoch must be divisible by WORLD_SIZE to avoid padded repeats')
        if self.num_samples <= 0 or (max_repeats is not None and max_repeats <= 0):
            raise ValueError('Sampling counts must be positive')
        if max_repeats is not None and self.num_samples > len(dataset) * max_repeats:
            raise ValueError('samples_per_epoch exceeds the per-row repeat capacity')
        self.max_repeats = max_repeats
        self.seed, self.epoch = seed, 0

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        # A finite ticket pool gives an explicit repeat cap. Without a cap,
        # multinomial sampling preserves the requested probabilities exactly.
        weights = self.weights if self.max_repeats is None else self.weights.repeat(self.max_repeats)
        indices = torch.multinomial(weights, self.num_samples,
                                    replacement=self.max_repeats is None, generator=generator)
        indices = (indices % len(self.weights)).tolist()
        if int(os.environ.get('RANK', '0')) == 0:
            counts = Counter(indices)
            print(json.dumps(dict(event='sft_sampling', epoch=self.epoch,
                                  draws=len(indices), unique_rows=len(counts),
                                  max_repeats=max(counts.values()),
                                  tasks=dict(Counter(self.tasks[i] for i in indices))), sort_keys=True), flush=True)
        return iter(indices)


class LengthGroupedSampler(Sampler):
    """按长度分组的确定性 sampler:同一 global step 的样本长度相近。

    每个 epoch 用 seed+epoch 洗牌,切成 mega_batch 大小的段,段内按长度降序;Accelerate 把
    连续的 per-device batch 轮流分给各 rank,所以一个 global step 拿到的是排序后连续的
    world_size 条,长度相近,段与段之间仍随机。每个 epoch 每行恰好出现一次,总步数、
    lr schedule 都不变,只改每一步的样本组合。按 turn 一行时随机分 batch 每步最长行平均
    11.5 万 token、行均值 3.8 万(2026-09-11 全量语料实测,4 node 100 s/it);分组后 27 s/it。
    长度用 length preflight 的精确渲染 token 数。
    """

    def __init__(self, lengths, *, seed, world_size=None, steps_per_group=32):
        self.lengths = [int(x) for x in lengths]
        if not self.lengths or min(self.lengths) < 0:
            raise ValueError('length_grouped sampling needs a non-negative length for every row')
        world = int(os.environ.get('WORLD_SIZE', '1')) if world_size is None else int(world_size)
        self.mega_batch_size = max(world * int(steps_per_group), 64)
        self.seed, self.epoch = int(seed), 0

    def __len__(self):
        return len(self.lengths)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.randperm(len(self.lengths), generator=generator).tolist()
        mb = self.mega_batch_size
        megabatches = [indices[i:i + mb] for i in range(0, len(indices), mb)]
        megabatches = [sorted(m, key=lambda i: self.lengths[i], reverse=True) for m in megabatches]
        order = [i for m in megabatches for i in m]
        if int(os.environ.get('RANK', '0')) == 0:
            print(json.dumps(dict(event='length_grouped_sampler', epoch=self.epoch, rows=len(order),
                                  mega_batch=mb, seed=self.seed,
                                  first_group_max=self.lengths[order[0]],
                                  first_group_min=self.lengths[order[min(mb, len(order)) - 1]]),
                             sort_keys=True), flush=True)
        return iter(order)
