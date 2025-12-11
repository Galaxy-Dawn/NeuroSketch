from omegaconf import DictConfig
import torch
from pathlib import Path
from src.data_module.data_func import DataFunction
from src.data_module.compute_metrics import MetricsFactory
from src.data_module.collate_fn import DataCollatorFactory
from src.data_module.dataset import DatasetFactory
from src.data_module.data_func import register_data
import numpy as np


@register_data('duin')
def duin_channel_data(cfg: DictConfig):
    processed_dir: Path = Path(cfg.dir.processed_dir) / cfg.dataset.name / cfg.dataset.task

    if cfg.dataset.version is None:
        if cfg.dataset.selected_channel:
            dataset_path = processed_dir / f'{cfg.dataset.name}_{cfg.dataset.id}_selected_channel_all_data.pt'
        else:
            dataset_path = processed_dir / f'{cfg.dataset.name}_{cfg.dataset.id}_all_data.pt'
    else:
        if cfg.dataset.selected_channel:
            dataset_path = processed_dir / f'{cfg.dataset.name}_{cfg.dataset.id}_selected_channel_{cfg.dataset.version}_data.pt'
        else:
            dataset_path = processed_dir / f'{cfg.dataset.name}_{cfg.dataset.id}_{cfg.dataset.version}_data.pt'

    all_dataset = torch.load(dataset_path, weights_only=True, map_location='cpu')
    train_split_filename = processed_dir / f'{cfg.dataset.name}_{cfg.dataset.id}_{cfg.dataset.version}_train_split.npy'
    eval_split_filename = processed_dir / f'{cfg.dataset.name}_{cfg.dataset.id}_{cfg.dataset.version}_eval_split.npy'
    test_split_filename = processed_dir / f'{cfg.dataset.name}_{cfg.dataset.id}_{cfg.dataset.version}_test_split.npy'
    train_split, eval_split, test_split = np.load(train_split_filename), np.load(eval_split_filename), np.load(
        test_split_filename)

    train_dataset = DatasetFactory(cfg.dataset.name)([all_dataset[i] for i in train_split])
    eval_dataset = DatasetFactory(cfg.dataset.name)([all_dataset[i] for i in eval_split])
    test_dataset = DatasetFactory(cfg.dataset.name)([all_dataset[i] for i in test_split])
    data_collator = DataCollatorFactory(cfg.dataset.name)
    compute_metrics = MetricsFactory(cfg.dataset.name)

    return DataFunction(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        test_dataset=test_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )