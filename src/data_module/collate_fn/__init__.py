from typing import Callable, Dict
from transformers.data.data_collator import DataCollator
import os
from src.utils.aux_func import import_modules

DATA_COLLATOR_FACTORY: Dict[str, DataCollator] = {}


def DataCollatorFactory(data_name: str) -> DataCollator:
    data_collator = DATA_COLLATOR_FACTORY.get(data_name, None)
    if data_collator is None:
        print(f"{data_name} data collator is not implmentation, use simple data collator")
        data_collator = DATA_COLLATOR_FACTORY.get('simple')
    return data_collator


def register_data_collator(name: str) -> Callable:
    def register_data_collator_cls(cls):
        if name in DATA_COLLATOR_FACTORY:
            return DATA_COLLATOR_FACTORY[name]

        DATA_COLLATOR_FACTORY[name] = cls
        return cls
    return register_data_collator_cls


models_dir = os.path.dirname(__file__)
import_modules(models_dir, "src.data_module.collate_fn")