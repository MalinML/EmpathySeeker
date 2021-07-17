from typing import Any, Dict, List, Union

import pandas as pd
import torch
from torch import LongTensor
from torch.utils.data import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer
import time

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
    def __init__(self, tokenizer: PreTrainedTokenizer, data_path: str, max_len: int = 512,) -> None:
        self.max_len = max_len
        self.tokenizer = tokenizer
        loading_start = time.time()
        cols = ["text", "ir", "er", "ex"]
        data = pd.read_csv(data_path, usecols=cols)
        print(f"Loading took {time.time()-loading_start}")

        data["text"] = data["text"].str.lower()

        tok_start = time.time()
        self.text_input = data["text"].to_list()
        print(f"Tokenizer took {time.time() - tok_start}")

        mapper = lambda x: 2 if x >= 1.5 else (1 if x >= 0.5 else 0)
        ex = ["EX" + str(mapper(x)) for x in data["ex"].to_list()]
        er = ["ER" + str(mapper(x)) for x in data["er"].to_list()]
        ir = ["IR" + str(mapper(x)) for x in data["ir"].to_list()]
        self.labels = [labels["".join(x)] for x in zip(ex, er, ir)]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: Union[LongTensor, int, List[int]]) -> Dict[str, Any]:
        input_text = self.text_input[idx]
        tokenized_input = self.tokenizer(input_text, max_length=self.max_len, padding="max_length", truncation=True, add_special_tokens=True)[
            "input_ids"
        ]

        return {
            "input_ids": tokenized_input,
            "labels": self.labels[idx],
        }

