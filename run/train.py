import os
import sys
from thop import profile
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf, ListConfig
from transformers import (
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from src.utils.log import setup_logging
from src.utils.aux_func import count_parameters, replace_eval_with_test, print_detailed_parameters
from src.model_module.model import MODEL_FACTORY
from src.data_module.data_func import DataFactory
import torch
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl


class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, early_stopping_patience: int = 1, early_stopping_threshold: float = 0.0):
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.last_best_metric = None
        self.patience_counter = 0

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        metric = state.log_history[-1].get('eval_' + args.metric_for_best_model)
        if metric is None:
            return

        if self.last_best_metric is None:
            self.last_best_metric = metric
            return

        improvement = metric - self.last_best_metric if args.greater_is_better else self.last_best_metric - metric
        if improvement > self.early_stopping_threshold:
            self.patience_counter = 0
            self.last_best_metric = metric
        else:
            self.patience_counter += 1

        if self.patience_counter >= self.early_stopping_patience:
            control.should_training_stop = True

@hydra.main(config_path="conf", config_name="train", version_base="1.2")
def main(cfg: DictConfig):
    LOGGER = setup_logging(level=20)
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{cfg.training.gpu_id}"

    training_args = TrainingArguments(
        seed=cfg.training.seed,
        do_train=cfg.training.do_train,
        do_eval=cfg.training.do_eval,
        do_predict=cfg.training.do_predict,
        overwrite_output_dir=cfg.training.overwrite_output_dir,
        output_dir=f'{cfg.dir.model_save_dir}/{cfg.wandb.exp_name}',
        learning_rate=cfg.training.learning_rate,
        warmup_ratio=cfg.training.warmup_ratio,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        weight_decay=cfg.training.weight_decay,
        dataloader_num_workers=cfg.training.dataloader_num_workers,
        num_train_epochs=cfg.training.num_train_epochs,
        eval_strategy=cfg.training.evaluation_strategy,
        # eval_steps=cfg.training.eval_steps,
        logging_dir=cfg.dir.logging_dir,
        logging_steps=cfg.training.logging_steps,
        save_strategy=cfg.training.save_strategy,
        # save_steps=cfg.training.save_steps,
        save_total_limit=cfg.training.save_total_limit,
        load_best_model_at_end=cfg.training.load_best_model_at_end,
        metric_for_best_model=cfg.training.metric_for_best_model,
        greater_is_better=cfg.training.greater_is_better,
        remove_unused_columns=cfg.training.remove_unused_columns,
        run_name=cfg.wandb.exp_name,
        dataloader_pin_memory=cfg.training.dataloader_pin_memory,
        save_safetensors=False,
    )
    wandb.init(
        project=cfg.wandb.project_name,
        name=f'{cfg.wandb.exp_name}',
        group=f'{cfg.wandb.group_name}',
        reinit=True,
        )
    wandb.config.update(OmegaConf.to_container(cfg, resolve=True))
    brain_model = MODEL_FACTORY.get(cfg.model.name)(cfg)
    data_module = DataFactory(cfg.dataset.name)(cfg)
    total_params = count_parameters(brain_model)
    print_detailed_parameters(brain_model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    brain_model.to(device)
    if isinstance(cfg.dataset.input_channels, ListConfig):
        eeg = torch.zeros((cfg.training.per_device_train_batch_size, cfg.dataset.input_channels[cfg.dataset.id-1],
                           cfg.dataset.seq_len[cfg.dataset.task]), device=device)
    else:
        eeg = torch.zeros((cfg.training.per_device_train_batch_size, cfg.dataset.input_channels,
                           cfg.dataset.seq_len[cfg.dataset.task]), device=device)
    labels = torch.zeros((cfg.training.per_device_train_batch_size,), device=device)
    flops, params = profile(brain_model, inputs=(eeg, labels))
    LOGGER.info_high(f"Total number of FLOPs in the model: {flops / 1e9} GFLOPs")
    LOGGER.info_high(f"Total number of parameters in the model: {total_params}")

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            LOGGER.info_high(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    set_seed(training_args.seed)

    trainer = Trainer(
        model=brain_model,
        args=training_args,
        train_dataset = data_module.train_dataset if training_args.do_train else None,
        eval_dataset = data_module.eval_dataset if training_args.do_eval else None,
        data_collator= data_module.data_collator,
        compute_metrics = data_module.compute_metrics,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=cfg.training.early_stopping_patience,
            early_stopping_threshold=0.001
        )]
    )

    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    if training_args.do_eval:
        metrics = trainer.evaluate()
        metrics['flops'] = flops
        metrics['params'] = params
        wandb.log(metrics)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        metrics = trainer.evaluate(data_module.test_dataset)
        metrics = replace_eval_with_test(metrics)
        metrics['flops'] = flops
        metrics['params'] = params
        wandb.log(metrics)
        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)


if __name__ == "__main__":
    main()