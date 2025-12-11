import numpy as np
import torch
from src.data_module.compute_metrics import register_metrics
from sklearn.metrics import top_k_accuracy_score
from sklearn.metrics import accuracy_score, f1_score, precision_score

@register_metrics('simple')
def simple_compute_metrics(eval_pred):
    logits, labels = eval_pred.predictions[1], eval_pred.predictions[0]
    labels = torch.tensor(labels)
    probs = torch.softmax(torch.tensor(logits), dim=1)
    preds = torch.argmax(probs, dim=1)
    top1_accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='weighted')
    f1 = f1_score(labels, preds, average='weighted')
    if isinstance(labels, np.ndarray):  # 检查是否为 numpy.ndarray
        labels = torch.tensor(labels)  # 转换为 torch.Tensor

    unique_labels = torch.unique(labels)
    class_accuracies = {}

    for label in unique_labels:
        true_positive = torch.sum((preds == label) & (labels == label)).item()
        total_samples = torch.sum(labels == label).item()
        if total_samples > 0:
            class_accuracies[f'acc_class_{label.item()}'] = true_positive / total_samples
        else:
            class_accuracies[f'acc_class_{label.item()}'] = 0.0

    return {
        # Model metrics
        'top1_accuracy'       : top1_accuracy,
        'precision'           : precision,
        'f1'                  : f1,
        # Class-specific metrics
        **class_accuracies
    }