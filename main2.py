from torch.utils.data.dataset import random_split
from posts_classifier.datasets.reddit import RedditDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import torch
from transformers import TrainingArguments, Trainer
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import EarlyStoppingCallback
from argparse import ArgumentTypeError
from posts_classifier.datasets.reddit import RedditDataset
from typing import Any, Tuple, Union

import pandas as pd
import torch
from tap.tap import Tap
from torch.utils.data import random_split
from torch.utils.data.dataloader import T
from torch.utils.data.dataset import Dataset
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer

TaskData = Tuple[PreTrainedTokenizer, PreTrainedModel, Dataset, Any]
from transformers import BertTokenizer
from typing import Any, Dict, List, Union

import pandas as pd
import torch
from torch import LongTensor
from torch.utils.data import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer


class Args(Tap):
    model_name: str = "bert-base-uncased"
    dataset: str = "data/small.csv"
    output_dir: str = "output/bert"
    cache_dir: str = "cached"
    batch_size: int = 8
    max_steps: int = -1
    warmup_steps: int = 0
    log_steps: int = 1000
    save_steps: int = 50000
    num_train_epochs: int = 5
    gradient_accumulation_steps: int = 1
    max_len: int = 64
    eval_steps: int = 300


args = Args().parse_args()
labels = [
    "EX0ER0IR0",
    "EX0ER0IR1",
    "EX0ER0IR2",
    "EX0ER1IR0",
    "EX0ER1IR1",
    "EX0ER1IR2",
    "EX0ER2IR0",
    "EX0ER2IR1",
    "EX0ER2IR2",
    "EX1ER0IR0",
    "EX1ER0IR1",
    "EX1ER0IR2",
    "EX1ER1IR0",
    "EX1ER1IR1",
    "EX1ER1IR2",
    "EX1ER2IR0",
    "EX1ER2IR1",
    "EX1ER2IR2",
    "EX2ER0IR0",
    "EX2ER0IR1",
    "EX2ER0IR2",
    "EX2ER1IR0",
    "EX2ER1IR1",
    "EX2ER1IR2",
    "EX2ER2IR0",
    "EX2ER2IR1",
    "EX2ER2IR2",
]
labels = {x: i for i, x in enumerate(labels)}

model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=27)


class RedditDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, data_path: str, max_len: int = 512,) -> None:
        self.tokenizer = tokenizer
        data = pd.read_csv(data_path).dropna()
        text_input = [
            tokenizer(x.lower(), max_length=max_len, padding="max_length", truncation=True, add_special_tokens=True, return_token_type_ids=True,)
            for x in data["text"].to_list()
        ]
        self.input_ids = [input["input_ids"] for input in text_input]
        mapper = lambda x: 2 if x >= 1.5 else (1 if x >= 0.5 else 0)
        ex = ["EX" + str(mapper(x)) for x in data["ex"].to_list()]
        er = ["ER" + str(mapper(x)) for x in data["er"].to_list()]
        ir = ["IR" + str(mapper(x)) for x in data["ir"].to_list()]
        self.labels = [labels["".join(x)] for x in zip(ex, er, ir)]

        # self.labels = pd.get_dummies(self.labels).values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: Union[LongTensor, int, List[int]]) -> Dict[str, Any]:
        return {
            "input_ids": self.input_ids[idx],
            "labels": self.labels[idx],
        }


dataset = RedditDataset(tokenizer, args.dataset, max_len=126)
dataset_size = len(dataset)  # type: ignore
train_size = int(0.7 * dataset_size)
eval_size = int(0.1 * dataset_size)
test_size = dataset_size - (train_size + eval_size)
train_dataset, eval_dataset, test_dataset = random_split(dataset, [train_size, eval_size, test_size])


def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred, average="macro")
    precision = precision_score(y_true=labels, y_pred=pred, average="macro")
    f1 = f1_score(y_true=labels, y_pred=pred, average="macro")

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


args = TrainingArguments(
    output_dir="output",
    evaluation_strategy="epoch",
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    num_train_epochs=args.num_train_epochs,
    seed=0,
    load_best_model_at_end=True,
)
trainer = Trainer(model=model, args=args, train_dataset=train_dataset, eval_dataset=eval_dataset, compute_metrics=compute_metrics,)

trainer.train()
