from typing import Dict
import os
from src.utils.aux_func import import_modules


DATASET_FACTORY: Dict = {}

def DatasetFactory(data_name: str):
    dataset = DATASET_FACTORY.get(data_name, None)
    if dataset is None:
        print(f"{data_name} metrics is not implmentation, use simple datasets")
        dataset = DATASET_FACTORY.get('simple')
    return dataset


def register_dataset(name: str):
    def register_dataset_cls(cls):
        if name in DATASET_FACTORY:
            return DATASET_FACTORY[name]
        DATASET_FACTORY[name] = cls
        return cls
    return register_dataset_cls


models_dir = os.path.dirname(__file__)
import_modules(models_dir, "src.data_module.dataset")