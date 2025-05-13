import logging
import math
import os
import re
from typing import Any, Callable, Optional
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm.auto import tqdm, trange

from transformers import Trainer
from transformers import PreTrainedModel
#from transformers import is_wandb_available
from transformers import TrainingArguments
from transformers.data.data_collator import DataCollator
from transformers import AdamW, get_linear_schedule_with_warmup
import wandb


logger = logging.getLogger(__name__)


class IETrainer(Trainer):
    """
    IETrainer is inheritated  from transformers.Trainer, optimized for IE tasks.
    """
    def __init__(
        self,
        model: PreTrainedModel,
        args: TrainingArguments,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        compute_metrics=None,
        tb_writer: Optional["SummaryWriter"] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = None,
        warmup_ratio: float = 0.1,
        best_f1: float = 0.0,
        best_step : int = 0,
        use_crf: Optional[bool]=False
    ):
        super(IETrainer, self).__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            optimizers=optimizers
        )
        self.best_f1 = best_f1
        self.best_step = best_step
        self.warmup_ratio = warmup_ratio
        self.use_crf = use_crf

    def get_optimizers(
        self,
        num_training_steps: int
    ) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
        """
        Setup the optimizer and the learning rate scheduler.
        """
        if self.optimizers is not None:
            return self.optimizers

        no_decay = ["bias", "LayerNorm.weight"]
        if self.use_crf:
            crf = "crf"
            crf_lr = self.args.crf_learning_rate
            logger.info(f"Learning rate for CRF: {crf_lr}")
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in self.model.named_parameters()
                        if (not any(nd in n for nd in no_decay)) and (crf not in n)
                    ],
                    "weight_decay": self.args.weight_decay
                },
                {
                    "params": [p for p in self.model.crf.parameters()],
                    "weight_decay": self.args.weight_decay,
                    "lr": crf_lr
                },
                {
                    "params": [
                        p for n, p in self.model.named_parameters()
                        if any(nd in n for nd in no_decay) and (not crf not in n)
                    ],
                    "weight_decay": 0.0,
                },
            ]
        else:
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in self.model.named_parameters()
                        if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in self.model.named_parameters()
                        if any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.0,
                },
            ]

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate,
            eps=self.args.adam_epsilon
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=num_training_steps
        )

        return optimizer, scheduler

    def evaluate(self, eval_dataset: Optional[Dataset] = None) -> Dict:
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        output = self._prediction_loop(eval_dataloader, description="Evaluation")
        step_f1 = output['metrics']['eval_f1']
        if step_f1 > self.best_f1:
            self.best_f1 = step_f1
            self.best_step = self.global_step
        logger.info(f"best_f1: {self.best_f1}, best_step: {self.best_step}")
        metrics = output['metrics']
        #del metrics['eval_entity_report']
        self._log(metrics)
        #print(output['metrics'])
        return output

    def predict(self, test_dataset: Dataset) -> Dict:
        test_dataloader = self.get_test_dataloader(test_dataset)
        # output = self._prediction_loop(test_dataloader, description="Prediction")
        # self._log(output['metrics'])
        return self._prediction_loop(test_dataloader, description="Prediction")

    def _prediction_loop(
        self,
        dataloader: DataLoader,
        description: str,
        metric_key_prefix: str = "eval",
    ) -> Dict:
        """
        Prediction/evaluation loop, shared by `evaluate()` and `predict()`
        Works both with or without labels.
        """
        model = self.model
        batch_size = dataloader.batch_size

        logger.info("***** Running %s *****", description)
        logger.info("  Num examples = %d", self.num_examples(dataloader))
        logger.info("  Batch size = %d", batch_size)

        model.eval()

        eval_losses: List[float] = []
        preds_ids = []
        label_ids = []

        for inputs in tqdm(dataloader, desc=description):
            has_labels = any(
                inputs.get(k) is not None
                for k in ["labels", "lm_labels", "masked_lm_labels"]
            )

            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(self.args.device)

            with torch.no_grad():
                outputs = model(**inputs)
                if has_labels:
                    step_eval_loss, logits = outputs[:2]
                    eval_losses += [step_eval_loss.mean().item()]
                else:
                    logits = outputs[0]

            mask = inputs["decoder_mask"].to(torch.bool)
            preds = model.decode(logits, mask=mask)
            preds_ids.extend(preds)
            if inputs.get("labels") is not None:
                labels = [inputs["labels"][i, mask[i]].tolist() \
                            for i in range(inputs["labels"].shape[0])]
                label_ids.extend(labels)
                assert len(preds) == len(labels)
                assert len(preds[0]) == len(labels[0])

        if self.compute_metrics is not None and \
                len(preds_ids) > 0 and \
                len(label_ids) > 0:
            metrics = self.compute_metrics(preds_ids, label_ids)
        else:
            metrics = {}
        if len(eval_losses) > 0:
            metrics['eval_loss'] = np.mean(eval_losses)

        # Prefix all keys with eval_
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return {'predictions': preds_ids, 'label_ids': label_ids, 'metrics': metrics}

    def _log(self, logs: Dict[str, float], iterator: Optional[tqdm] = None) -> None:
        if self.epoch is not None:
            logs["epoch"] = self.epoch
        if self.global_step is None:
            # when logging evaluation metrics without training
            self.global_step = 0
        #if is_wandb_available():
        if self.is_world_master():
            wandb.log(logs, step=self.global_step)
        output = {**logs, **{"step": self.global_step}}
        if iterator is not None:
            iterator.write(output)
        else:
            logger.info(
                {k:round(v, 4) if isinstance(v, float) else v for k, v in output.items()}
            )

