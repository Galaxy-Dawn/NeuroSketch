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


def gather_data(data_args, pad = False):
    eeg_list = []
    mask_list = []
    label_list = []
    id = data_args.dataset.id
    data_path = f'/data/share/data/ThingsEEG/sub-{id:02d}'
    file_names = os.listdir(data_path)
    if data_args.dataset.task == 'test':
        file_names = [f for f in file_names if 'test' in f]
    elif data_args.dataset.task == 'train':
        file_names = [f for f in file_names if 'train' in f]
    else:
        raise ValueError('Task must be train or test')
    for file in tqdm(file_names, desc='Loading data'):
        total_data = np.load(data_path + '/' + file, allow_pickle=True).item()
        eeg = total_data['preprocessed_eeg_data']
        if data_args.dataset.task == 'test':
            label = np.arange(eeg.shape[0])[:, np.newaxis] * np.ones(eeg.shape[1], dtype=int)
            eeg = einops.rearrange(eeg, 't r c l -> (t r) c l')
            label = einops.rearrange(label, 't r -> (t r)')
        elif data_args.dataset.task == 'train':
            class_indices = np.repeat(np.arange(1654), 10)
            label = np.tile(class_indices[:, np.newaxis], (1, 4))
            eeg = einops.rearrange(eeg, 't r c l -> (t r) c l')
            label = einops.rearrange(label, 't r -> (t r)')
        save_index = check_ieeg(eeg)
        eeg = eeg[save_index]
        label = label[save_index]
        if pad:
            pad_size = data_args.dataset.input_channels - eeg.shape[1]
            if pad_size > 0:
                padded_ieeg = np.pad(eeg, ((0, 0), (0, pad_size),(0, 0)), mode='constant', constant_values=0)
                mask = np.ones((eeg.shape[0], data_args.dataset.input_channels))
                mask[:, eeg.shape[1]:] = 0
            else:
                padded_ieeg = eeg[:, :data_args.dataset.input_channels]
                mask = np.ones((eeg.shape[0], data_args.dataset.input_channels))
        else:
            padded_ieeg = eeg
            mask = np.ones((eeg.shape[0], eeg.shape[1]))

        eeg_list.append(padded_ieeg)
        mask_list.append(mask)
        label_list.append(label)

    eeg = np.concatenate(eeg_list, axis=0)
    mask = np.concatenate(mask_list, axis=0)
    label = np.concatenate(label_list, axis=0)

    return eeg, mask, label


@hydra.main(config_path="../conf", config_name="prepare_data", version_base="1.2")
def prepare_data(cfg: DictConfig):
    LOGGER = setup_logging(level = 20)
    processed_dir: Path = Path(cfg.dir.processed_dir) / cfg.dataset.name / cfg.dataset.task
    processed_dir.mkdir(parents=True, exist_ok=True)
    # if processed_dir.exists():
    #     shutil.rmtree(processed_dir)
    #     LOGGER.info_high(f"Removed dir: {processed_dir}")
    pad = cfg.dataset.pad
    id = cfg.dataset.id
    with tracking("Load and gather data", LOGGER):
        ieeg, mask, label = gather_data(cfg, pad)
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
                'ieeg_mask'    : torch.tensor(mask[i]),
                'labels'       : torch.tensor(label[i]),
            }
            dataset_list.append(data_dict)
        torch.save(dataset_list, processed_dir / f'{cfg.dataset.name}_{id}_all_data.pt')

if __name__ == '__main__':
    prepare_data()