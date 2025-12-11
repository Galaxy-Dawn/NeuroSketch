from typing import Callable, Dict, Optional, Union
from transformers.trainer_utils import EvalPrediction
import os
from src.utils.aux_func import import_modules


METRICS_FACTORY: Dict[str, Callable[[EvalPrediction], Dict]] = {}

def MetricsFactory(data_name: str) -> Callable[[EvalPrediction], Dict]:
    metrics = METRICS_FACTORY.get(data_name, None)
    if metrics is None:
        print(f"{data_name} metrics is not implmentation, use simple metrics")
        metrics = METRICS_FACTORY.get('simple')
    return metrics


def register_metrics(name: str) -> Callable:
    def register_metrics_cls(cls):
        if name in METRICS_FACTORY:
            return METRICS_FACTORY[name]

        METRICS_FACTORY[name] = cls
        return cls
    return register_metrics_cls


models_dir = os.path.dirname(__file__)
import_modules(models_dir, "src.data_module.compute_metrics")