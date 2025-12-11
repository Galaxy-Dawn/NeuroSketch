import sys
from omegaconf import DictConfig
import hydra
import einops
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


def get_duin_info(data_path):
    data_info = []
    for folder in os.listdir(data_path):
        if folder.isdigit():
            folder_path = os.path.join(data_path, folder)
            if os.path.isdir(folder_path):
                data_info.append({
                    'id': folder,
                    'data_path': os.path.join(folder_path, 'data.npy'),
                    'data_after_norm_path': os.path.join(folder_path, 'data_after_norm.npy'),
                    'data_after_aug_path': os.path.join(folder_path, 'data_after_aug.npy'),
                    'delta_path': os.path.join(folder_path, 'data_delta.npy'),
                    'theta_path': os.path.join(folder_path, 'data_theta.npy'),
                    'alpha_path': os.path.join(folder_path, 'data_alpha.npy'),
                    'beta_path': os.path.join(folder_path, 'data_beta.npy'),
                    'low_gamma_path': os.path.join(folder_path, 'data_low gamma.npy'),
                    'high_gamma_path': os.path.join(folder_path, 'data_high gamma.npy'),
                    'label_path': os.path.join(folder_path, 'label.npy'),
                    'text_path': os.path.join(folder_path, 'word.npy'),
                    'channel_path': os.path.join(folder_path, 'selected_channel_index.npy')
                })
    df = pd.DataFrame(data_info)
    df['id'] = df['id'].astype(int)
    df = df.sort_values(by='id').reset_index(drop=True)
    return df


def gather_data(data_args, data_info, version = None, selected_channel = False, pad = False):
    ieeg_list = []
    mask_list = []
    label_list = []
    text_list = []
    for i in tqdm(range(len(data_info)), desc='Loading data'):
        label_path, text = data_info.iloc[i]['label_path'], data_info.iloc[i]['text_path']
        ieeg_path = data_info.iloc[i]['data_path'] if version is None else data_info.iloc[i][version + '_path']
        ieeg = np.load(ieeg_path, allow_pickle=True)
        if selected_channel:
            choose_channel_index = np.load(data_info.iloc[i]['channel_path'], allow_pickle=True)
            ieeg = einops.rearrange(ieeg, 'c b t -> b c t')
            ieeg = ieeg[:, choose_channel_index]
        else:
            ieeg = einops.rearrange(ieeg, 'c b t -> b c t')

        save_index = check_ieeg(ieeg)
        ieeg = ieeg[save_index]
        label = np.load(label_path, allow_pickle=True)
        label = label[save_index]
        text = np.load(text, allow_pickle=True)
        text = text[save_index]
        assert len(ieeg) == len(label) == len(text), "Lengths of ieeg, label, and text do not match"
        if pad:
            pad_size = data_args.dataset.input_channels - ieeg.shape[1]
            if pad_size > 0:
                padded_ieeg = np.pad(ieeg, ((0, 0), (0, pad_size),(0, 0)), mode='constant', constant_values=0)
                mask = np.ones((ieeg.shape[0], data_args.dataset.input_channels))
                mask[:, ieeg.shape[1]:] = 0
            else:
                padded_ieeg = ieeg[:, :data_args.dataset.input_channels]
                mask = np.ones((ieeg.shape[0], data_args.dataset.input_channels))
        else:
            padded_ieeg = ieeg
            mask = np.ones((ieeg.shape[0], ieeg.shape[1]))

        ieeg_list.append(padded_ieeg)
        mask_list.append(mask)
        label_list.append(label)
        text_list.append(text)

    ieeg = np.concatenate(ieeg_list, axis=0)
    mask = np.concatenate(mask_list, axis=0)
    label = np.concatenate(label_list, axis=0)
    text = np.concatenate(text_list, axis=0)

    return ieeg, mask, label, text


@hydra.main(config_path="../conf", config_name="prepare_data", version_base="1.2")
def prepare_data(cfg: DictConfig, ctc = False, ctc_dict = None, ctc_dict_with_tone = None):
    LOGGER = setup_logging(level = 20)
    processed_dir: Path = Path(cfg.dir.processed_dir) / cfg.dataset.name / cfg.dataset.task
    processed_dir.mkdir(parents=True, exist_ok=True)
    # if processed_dir.exists():
    #     shutil.rmtree(processed_dir)
    #     LOGGER.info_high(f"Removed dir: {processed_dir}")
    version = cfg.dataset.version
    selected_channel = cfg.dataset.selected_channel
    pad = cfg.dataset.pad
    id = cfg.dataset.id
    data_info_csv = pd.read_csv(Path(cfg.dir.data_dir) / cfg.dataset.name / cfg.dataset.task / 'data_info.csv')
    data_info = data_info_csv.loc[data_info_csv['id'] == id]
    with tracking("Load and gather data", LOGGER):
        ieeg, mask, label, text = gather_data(cfg, data_info, version = version, selected_channel=selected_channel, pad = pad)
        LOGGER.info_high(f"Loaded data with shape: {ieeg.shape}")


    with tracking("Get and save split", LOGGER):
        if cfg.split_method == 'simple':
            train_split, eval_split, test_split = get_split(cfg, ieeg, label=label)

            if version is None:
                train_split_filename = f'{cfg.dataset.name}_{id}_train_split.npy'
                eval_split_filename = f'{cfg.dataset.name}_{id}_eval_split.npy'
                test_split_filename = f'{cfg.dataset.name}_{id}_test_split.npy'
            else:
                train_split_filename = f'{cfg.dataset.name}_{id}_{version}_train_split.npy'
                eval_split_filename = f'{cfg.dataset.name}_{id}_{version}_eval_split.npy'
                test_split_filename = f'{cfg.dataset.name}_{id}_{version}_test_split.npy'

            np.save(processed_dir / train_split_filename, train_split)
            np.save(processed_dir / eval_split_filename, eval_split)
            np.save(processed_dir / test_split_filename, test_split)

        elif cfg.split_method == 'n_fold':
            fold_splits, test_indices = get_n_fold_split(cfg, ieeg, label=label)
            for fold_idx, (train_split, eval_split) in enumerate(fold_splits):
                if version is None:
                    train_split_filename = f'{cfg.dataset.name}_{id}_fold{fold_idx}_train_split.npy'
                    eval_split_filename = f'{cfg.dataset.name}_{id}_fold{fold_idx}_eval_split.npy'
                    test_split_filename = f'{cfg.dataset.name}_{id}_fold{fold_idx}_test_split.npy'
                else:
                    train_split_filename = f'{cfg.dataset.name}_{id}_{version}_fold{fold_idx}_train_split.npy'
                    eval_split_filename = f'{cfg.dataset.name}_{id}_{version}_fold{fold_idx}_eval_split.npy'
                    test_split_filename = f'{cfg.dataset.name}_{id}_{version}_fold{fold_idx}_test_split.npy'

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
            if ctc:
                data_dict['ctc_labels'] = torch.tensor(ctc_dict[text[i]])
                data_dict['ctc_labels_with_tone'] = torch.tensor(ctc_dict_with_tone[text[i]])
            dataset_list.append(data_dict)

        if version is None:
            if selected_channel:
                torch.save(dataset_list, processed_dir / f'{cfg.dataset.name}_{id}_selected_channel_all_data.pt')
            else:
                torch.save(dataset_list, processed_dir / f'{cfg.dataset.name}_{id}_all_data.pt')
        else:
            if selected_channel:
                torch.save(dataset_list, processed_dir / f'{cfg.dataset.name}_{id}_selected_channel_{version}_data.pt')
            else:
                torch.save(dataset_list, processed_dir / f'{cfg.dataset.name}_{id}_{version}_data.pt')


if __name__ == '__main__':
    data_info = get_duin_info("/data/share/data/duin")
    os.makedirs("/data/share/data/duin/word_classification", exist_ok=True)
    data_info.to_csv("/data/share/data/duin/word_classification" + '/data_info.csv')
    prepare_data()
