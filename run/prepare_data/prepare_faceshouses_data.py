import scipy.io
import torch
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

def split_epochs(mat_data):
    """
    根据 stim 分割数据为不同事件段
    Returns:
        list of dict: [{
            'start_idx': 起始索引,
            'end_idx': 结束索引,
            'label': 'face/house/other'
        }, ...]
    """
    stim = mat_data['stim'].squeeze().astype(int)
    events = []
    current_start = None
    current_label = None

    for i, val in enumerate(stim):
        # 确定标签逻辑
        if 1 <= val <= 50:
            label = 'house'
        elif 51 <= val <= 100:
            label = 'face'
        else:
            label = 'other'

        # 边界检测逻辑
        if val in [0, 101]:
            if current_start is not None:
                # 保存前一个有效区间
                events.append({
                    'start_idx': current_start,
                    'end_idx'  : i - 1,
                    'label'    : current_label
                })
                current_start = None
        else:
            if current_start is None:
                current_start = i
            current_label = label  # 持续更新标签

    # 处理最后一个未闭合的区间
    if current_start is not None:
        events.append({
            'start_idx': current_start,
            'end_idx'  : len(stim) - 1,
            'label'    : current_label
        })

    return [e for e in events if e['label'] in ['face', 'house']]


def process_eeg_data(file_path):
    """
    处理EEG数据并生成结构化数据集
    Returns:
        eeg_data: np.ndarray [n_epochs, n_channels, n_timesteps]
        labels: np.ndarray [n_epochs,]
    """
    mat_data = scipy.io.loadmat(file_path)
    # 分割事件段
    epochs = split_epochs(mat_data)
    # 初始化数据容器
    eeg_data = []
    labels = []
    # 提取原始EEG数据（假设数据存储在mat的'data'字段）
    raw_data = mat_data['data']
    for epoch in epochs:
        epoch_data = raw_data[epoch['start_idx']:epoch['end_idx'] + 1, :]
        eeg_data.append(epoch_data.T)
        label = 0 if epoch['label'] == 'face' else 1
        labels.append(label)

    # 将列表转换为numpy数组
    eeg_data = np.stack(eeg_data)  # 最终形状 [batch, channels, timesteps]
    labels = np.array(labels)

    return eeg_data, labels


def gather_data(data_args, pad = False):
    subject_id_map = ["aa", "ap", "ca", "de", "fp", "ha", "ja", "jm", "jt", "mv", "rn", "rr", "wc", "zt"]
    task = data_args.dataset.task
    id = data_args.dataset.id
    data_path = '/data/share/data/Stanford_ecog/faces_basic/data'
    eeg, label = process_eeg_data(f'{data_path}/{subject_id_map[id]}/{subject_id_map[id]}_{task}.mat')
    return eeg, label


@hydra.main(config_path="../conf", config_name="prepare_data", version_base="1.2")
def prepare_data(cfg: DictConfig):
    LOGGER = setup_logging(level = 20)
    processed_dir: Path = Path(cfg.dir.processed_dir) / cfg.dataset.name / cfg.dataset.task
    processed_dir.mkdir(parents=True, exist_ok=True)
    # if processed_dir.exists():
    #     shutil.rmtree(processed_dir)
    #     LOGGER.info_high(f"Removed dir: {processed_dir}")
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