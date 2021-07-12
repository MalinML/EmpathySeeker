from typing import Any, Dict, List, Union

import pandas as pd
import torch
from torch import LongTensor
from torch.utils.data import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer

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


class RedditDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, data_path: str, logger, max_len: int = 512,) -> None:
        logger.info("Loading dataset")
        self.tokenizer = tokenizer
        data = pd.read_csv(data_path).dropna()
        for x in data["text"].to_list():
            try:
                x.lower()
            except:
                print(x)
        text_input = [
            tokenizer(x.lower(), max_length=max_len, padding="max_length", truncation=True, add_special_tokens=True, return_token_type_ids=True,)
            for x in data["text"].to_list()
        ]
        self.input_ids = [input["input_ids"] for input in text_input]
        mapper = lambda x: 2 if x >= 1.5 else (1 if x >= 0.5 else 0)
        ex = ["EX" + str(mapper(x)) for x in data["ex"].to_list()]
        er = ["ER" + str(mapper(x)) for x in data["er"].to_list()]
        ir = ["IR" + str(mapper(x)) for x in data["ir"].to_list()]
        self.labels = ["".join(x) for x in zip(ex, er, ir)]

        self.labels = pd.get_dummies(self.labels).values
        print("We have ", len(self.labels[0]), "labels")
        logger.info("Dataset loaded")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: Union[LongTensor, int, List[int]]) -> Dict[str, Any]:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return {
            "input_ids": self.input_ids[idx],
            "labels": self.labels[idx],
        }


# Collate function
def create_collate():
    def collate_fn(batch):
        # Inputs
        input_ids = torch.LongTensor([f["input_ids"] for f in batch])
        # Labels
        labels = torch.FloatTensor([f["labels"] for f in batch])

        return input_ids, labels

    return collate_fn
