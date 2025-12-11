from typing import Callable, Dict
from torch import nn
import os
from src.utils.aux_func import import_modules

MODEL_FACTORY: Dict[str, nn.Module] = {}

def ModelFactory(model_name: str) -> nn.Module:
    model = MODEL_FACTORY.get(model_name, None)
    assert model, f"{model_name} model is not implmentation"
    return model

def register_model(name: str) -> Callable:
    def register_model_cls(cls):
        if name in MODEL_FACTORY:
            return MODEL_FACTORY[name]
        MODEL_FACTORY[name] = cls
        return cls
    return register_model_cls

models_dir = os.path.dirname(__file__)
import_modules(models_dir, "src.model_module.model")