import torch
from typing import Sequence, Dict
from src.data_module.collate_fn import register_data_collator


@register_data_collator("simple")
def simple_collate_fn(instances: Sequence[Dict])-> Dict[str, torch.Tensor]:
    ieeg_raw_data, labels = tuple([instance[key] for instance in instances]
                                 for key in ("ieeg_raw_data", "labels"))
    ieeg_raw_data = torch.stack(ieeg_raw_data)
    labels = torch.tensor(labels)

    batch = {
        "ieeg_raw_data": ieeg_raw_data,
        "labels"       : labels,
        "return_loss"  : True,
    }
    return batch

