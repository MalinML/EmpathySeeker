from enum import Enum
from posts_classifier.models.classifier import get_all_labels_classifier
from posts_classifier.utils import get_logger, set_seed
from torch.utils.data.dataset import random_split
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import torch
from transformers import TrainingArguments, Trainer, EvalPrediction
from typing import Any, Optional, Tuple, Union

import pandas as pd
import torch
from tap.tap import Tap
from torch.utils.data import random_split
from torch.utils.data.dataloader import T
from torch.utils.data.dataset import Dataset
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer

TaskData = Tuple[PreTrainedTokenizer, PreTrainedModel, Dataset, Any]
from typing import Any, Dict, List, Union

import pandas as pd
import torch
from torch import LongTensor
from torch.utils.data import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer


logger = get_logger("main.log")  # Logger


class DatasetType(Enum):
    ALL_LABELS = 1  # all 26 labels


class Args(Tap):
    model_name: str = "bert-base-uncased"
    dataset: str = "data/small.csv"
    output_dir: str = "output/bert"
    cache_dir: str = "cached"
    batch_size: int = 8
    max_steps: int = -1
    warmup_steps: int = 0
    num_train_epochs: int = 5
    do_train: bool = False
    do_eval: bool = False
    do_test: bool = False
    data_type: Optional[DatasetType] = None


set_seed()


def compute_metrics(p: EvalPrediction):
    # TODO: load metrics like https://github.com/huggingface/transformers/blob/master/examples/pytorch/text-classification/run_glue.py
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred, average="macro")
    precision = precision_score(y_true=labels, y_pred=pred, average="macro")
    f1 = f1_score(y_true=labels, y_pred=pred, average="macro")

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def get_dataset(type: Optional[DatasetType], tokenizer, data_path):
    if type == DatasetType.ALL_LABELS:
        from posts_classifier.datasets.reddit_all_classes import RedditDataset

        dataset = RedditDataset(tokenizer, data_path)
    else:
        raise ValueError(f"Unknown data type {type}")
    dataset_size = len(dataset)  # type: ignore
    train_size = int(0.7 * dataset_size)
    eval_size = int(0.1 * dataset_size)
    test_size = dataset_size - (train_size + eval_size)
    train_dataset, eval_dataset, test_dataset = random_split(dataset, [train_size, eval_size, test_size])
    return train_dataset, eval_dataset, test_dataset


def main():
    global logger
    args = Args().parse_args()

    logger = get_logger("{}.log".format(args.model_name.split("/")[-1]))  # Logger
    model, tokenizer = get_all_labels_classifier(args.model_name)
    train_dataset, eval_dataset, test_dataset = get_dataset(args.data_type, tokenizer, args.dataset)
    training_args = TrainingArguments(
        output_dir="output",
        evaluation_strategy="epoch",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        seed=0,
        load_best_model_at_end=True,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset, compute_metrics=compute_metrics,)
    if args.do_train:
        print("*** Train ***")
        trainer.train()
    if args.do_eval:
        print("*** Eval ***")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        print(metrics)

    print("ALL DONE")

