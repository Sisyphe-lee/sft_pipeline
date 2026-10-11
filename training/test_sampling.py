from collections import Counter

from accelerate.data_loader import prepare_data_loader
from datasets import Dataset
import pytest
import torch
from torch.utils.data import DataLoader

from training.sampling import EpisodeSampler, LengthGroupedSampler


def dataset():
    return Dataset.from_dict(dict(task_id=['a'] * 50 + ['b'] * 50,
                                  trajectory_id=['short'] * 10 + ['long'] * 40 + ['only'] * 50))


def test_episode_sampling_probabilities_epoch_and_repeat_cap():
    data = dataset()
    sampler = EpisodeSampler(data, strategy='episode_uniform', seed=19, num_samples=12000)
    indices = list(sampler)
    episodes = Counter(data[int(i)]['trajectory_id'] for i in indices)
    assert all(abs(count / len(indices) - 1/3) < .025 for count in episodes.values())
    assert list(sampler) == indices
    sampler.set_epoch(1)
    assert list(sampler) != indices
    resumed = EpisodeSampler(data, strategy='episode_uniform', seed=19, num_samples=12000)
    resumed.set_epoch(1)
    assert list(resumed) == list(sampler)
    capped = EpisodeSampler(data, strategy='task_episode_uniform', seed=19, num_samples=150, max_repeats=2)
    assert max(Counter(capped).values()) == 2
    balanced = EpisodeSampler(data, strategy='task_episode_uniform', seed=19, num_samples=12000)
    tasks = Counter(data[int(i)]['task_id'] for i in balanced)
    assert abs(tasks['a'] / 12000 - .5) < .025


def test_accelerate_shards_one_stream_and_forwards_epoch(monkeypatch):
    monkeypatch.setenv('WORLD_SIZE', '2')
    data = dataset()
    expected = EpisodeSampler(data, strategy='episode_uniform', seed=19, num_samples=100)
    expected.set_epoch(3)
    shards = []
    for rank in range(2):
        sampler = EpisodeSampler(data, strategy='episode_uniform', seed=19, num_samples=100)
        loader = DataLoader(list(range(len(data))), batch_size=1, sampler=sampler)
        sharded = prepare_data_loader(loader, device=torch.device('cpu'), num_processes=2,
                                      process_index=rank, put_on_device=False)
        sharded.set_epoch(3)
        assert sampler.epoch == 3
        shards.append([int(batch[0]) for batch in sharded])
    assert [item for pair in zip(*shards) for item in pair] == list(expected)
    with pytest.raises(ValueError, match='divisible'):
        EpisodeSampler(data, strategy='episode_uniform', seed=19, num_samples=99)
    with pytest.raises(ValueError, match='repeat capacity'):
        EpisodeSampler(data, strategy='episode_uniform', seed=19, num_samples=202, max_repeats=2)


def test_length_grouped_sampler_groups_by_length_and_reshuffles(monkeypatch):
    monkeypatch.setenv('WORLD_SIZE', '2')
    lengths = list(range(1, 301))  # 300 行,长度 1..300
    sampler = LengthGroupedSampler(lengths, seed=7, steps_per_group=32)
    assert sampler.mega_batch_size == 64 and len(sampler) == 300
    order = list(iter(sampler))
    assert sorted(order) == list(range(300))  # 每行恰好一次
    for start in range(0, 300, 64):  # 段内按长度降序
        group = [lengths[i] for i in order[start:start + 64]]
        assert group == sorted(group, reverse=True)
    assert list(iter(sampler)) == order  # 同 epoch 确定性
    sampler.set_epoch(1)
    assert list(iter(sampler)) != order  # 换 epoch 重洗
    with pytest.raises(ValueError):
        LengthGroupedSampler([3, -1], seed=1)
