from omegaconf import DictConfig
import torch
from pathlib import Path
from src.data_module.data_func import DataFunction
from src.data_module.compute_metrics import MetricsFactory
from src.data_module.collate_fn import DataCollatorFactory
from src.data_module.dataset import DatasetFactory
from src.data_module.data_func import register_data
import numpy as np


@register_data('simple')
def simple_data(cfg: DictConfig):
    processed_dir: Path = Path(cfg.dir.processed_dir) / cfg.dataset.name / cfg.dataset.task
    dataset_path = processed_dir / f'{cfg.dataset.name}_{cfg.dataset.id}_all_data.pt'
    all_dataset = torch.load(dataset_path, weights_only=True, map_location='cpu')
    if cfg.dataset.split_method == 'simple':
        train_split_filename = processed_dir / f'{cfg.dataset.name}_{cfg.dataset.id}_train_split.npy'
        eval_split_filename = processed_dir / f'{cfg.dataset.name}_{cfg.dataset.id}_eval_split.npy'
        test_split_filename = processed_dir / f'{cfg.dataset.name}_{cfg.dataset.id}_test_split.npy'
        train_split, eval_split, test_split = np.load(train_split_filename), np.load(eval_split_filename), np.load(
            test_split_filename)
    elif cfg.dataset.split_method == 'n_fold':
        fold_idx = cfg.dataset.fold
        train_split_filename = processed_dir / f'{cfg.dataset.name}_{cfg.dataset.id}_fold{fold_idx}_train_split.npy'
        eval_split_filename = processed_dir / f'{cfg.dataset.name}_{cfg.dataset.id}_fold{fold_idx}_eval_split.npy'
        test_split_filename = processed_dir / f'{cfg.dataset.name}_{cfg.dataset.id}_fold{fold_idx}_test_split.npy'
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