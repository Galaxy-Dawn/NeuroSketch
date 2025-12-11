import torch.nn as nn

def get_act(act_type: str):
    if act_type == "ReLU":
        return nn.ReLU()
    elif act_type == "GELU":
        return nn.GELU()
    elif act_type == "SiLU":
        return nn.SiLU()
    elif act_type == "Mish":
        return nn.Mish()
    elif act_type == "HardSwish":
        return nn.Hardswish()
    else:
        raise ValueError(f"act_type {act_type} not supported")
