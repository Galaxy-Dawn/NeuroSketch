import sys
from omegaconf import DictConfig
import hydra
import einops
import json
from tqdm import tqdm
import pandas as pd
import numpy as np
from src.data_module.utils import check_ieeg, get_split, get_n_fold_split
import torch
from pathlib import Path
import os
from src.utils.log import setup_logging, cprint, tracking
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="torch")


def reshape_eeg(eeg, num_blocks=40, video_duration=2, rest_duration=3, sample_rate=200):
    block_length = (video_duration * 5 + rest_duration) * sample_rate
    reshaped_eeg = np.zeros((eeg.shape[0], eeg.shape[1], num_blocks * 5, video_duration * sample_rate))
    for block_idx in range(num_blocks):
        # 计算当前 block 的起始和结束索引
        start_idx = block_idx * block_length
        end_idx = start_idx + block_length
        # 提取当前 block 的数据
        block_data = eeg[:, :, start_idx:end_idx]
        # 提取 5 个 2 秒的视频部分
        for video_idx in range(5):
            video_start_idx = rest_duration * sample_rate + video_idx * (video_duration * sample_rate)
            video_end_idx = video_start_idx + (video_duration * sample_rate)
            video_data = block_data[:, :, video_start_idx:video_end_idx]
            # 将视频数据放入重新组织后的数组中
            reshaped_eeg[:, :, block_idx * 5 + video_idx, :] = video_data
    reshaped_eeg = einops.rearrange(reshaped_eeg, 'b c s p -> (b s) c p')
    return reshaped_eeg


def gather_data(data_args, pad = False):
    task = data_args.dataset.task
    id = data_args.dataset.id
    data_path = '/data/share/data/SEED-DV/EEG'
    eeg = np.load(f'{data_path}/sub{id}.npy', allow_pickle=True)
    eeg = reshape_eeg(eeg)
    if task == 'concept_classification':
        label = np.load('/data/share/data/SEED-DV/Video/meta-info/All_video_label.npy', allow_pickle=True)
        label = np.repeat(label, 5, axis=1)
        label -= 1
    label = label.flatten()
    return eeg, label


@hydra.main(config_path="../conf", config_name="prepare_data", version_base="1.2")
def prepare_data(cfg: DictConfig):
    LOGGER = setup_logging(level = 20)
    processed_dir: Path = Path(cfg.dir.processed_dir) / cfg.dataset.name / cfg.dataset.task
    processed_dir.mkdir(parents=True, exist_ok=True)
    pad = cfg.dataset.get('pad', False)
    id = cfg.dataset.id
    with tracking("Load and gather data", LOGGER):
        ieeg, label = gather_data(cfg, pad)
        LOGGER.info_high(f"Loaded data with shape: {ieeg.shape}")

    with tracking("Get and save split", LOGGER):
        if cfg.split_method == 'simple':
            train_split, eval_split, test_split = get_split(cfg, ieeg, label=label)

            train_split_filename = f'{cfg.dataset.name}_{id}_train_split.npy'
            eval_split_filename = f'{cfg.dataset.name}_{id}_eval_split.npy'
            test_split_filename = f'{cfg.dataset.name}_{id}_test_split.npy'

            np.save(processed_dir / train_split_filename, train_split)
            np.save(processed_dir / eval_split_filename, eval_split)
            np.save(processed_dir / test_split_filename, test_split)

        elif cfg.split_method == 'n_fold':
            fold_splits, test_indices = get_n_fold_split(cfg, ieeg, label=label)
            for fold_idx, (train_split, eval_split) in enumerate(fold_splits):
                train_split_filename = f'{cfg.dataset.name}_{id}_fold{fold_idx}_train_split.npy'
                eval_split_filename = f'{cfg.dataset.name}_{id}_fold{fold_idx}_eval_split.npy'
                test_split_filename = f'{cfg.dataset.name}_{id}_fold{fold_idx}_test_split.npy'

                np.save(processed_dir / train_split_filename, train_split)
                np.save(processed_dir / eval_split_filename, eval_split)
                np.save(processed_dir / test_split_filename, test_indices)

    with tracking("Prepare and save data", LOGGER):
        dataset_list = []
        for i in tqdm(range(len(ieeg)),desc='Preparing data'):
            data_dict = {
                'ieeg_raw_data': torch.tensor(ieeg[i]),
                'labels'       : torch.tensor(label[i]),
            }
            dataset_list.append(data_dict)
        torch.save(dataset_list, processed_dir / f'{cfg.dataset.name}_{id}_all_data.pt')


if __name__ == '__main__':
    prepare_data()




