from argparse import ArgumentTypeError
from posts_classifier.datasets.reddit import RedditDataset
from typing import Any, Tuple, Union

import pandas as pd
import torch
from coai_base.training.arguments import TrainerArguments
from coai_base.utils import get_logger, resolve_device, set_seed
from tap.tap import Tap
from torch.utils.data import random_split
from torch.utils.data.dataloader import T
from torch.utils.data.dataset import Dataset
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer

TaskData = Tuple[PreTrainedTokenizer, PreTrainedModel, Dataset, Any]


class Args(Tap, TrainerArguments):
    model_name: str = "bert-base-uncased"
    skip_train: bool = False
    skip_eval: bool = False
    skip_test: bool = False
    dataset: str = "data/small.csv"
    output_dir: str = "output/bert"
    cache_dir: str = "cached"
    batch_size: int = 8
    n_gpu: int = torch.cuda.device_count()
    device: Union[str, torch.device] = resolve_device()
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    seed: int = 42
    max_steps: int = -1
    warmup_steps: int = 0
    log_steps: int = 1000
    save_steps: int = 50000
    num_train_epochs: int = 5
    gradient_accumulation_steps: int = 1
    max_len: int = 64


args = Args().parse_args()

set_seed(args.seed, args.n_gpu)  # For reproducibility
logger = get_logger("{}.log".format(args.model_name.split("/")[-1]))  # Logger


def prepare_posts_classification() -> TaskData:
    from transformers import BertTokenizer

    from posts_classifier.datasets.reddit import RedditDataset, create_collate
    from posts_classifier.models.classifier import Bert

    # Tokenizer
    print("Loading", args.model_name, "to", args.cache_dir)
    tokenizer = BertTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)

    # Model
    model = Bert()
    model.to(args.device)

    # Dataset
    dataset = RedditDataset(tokenizer, args.dataset, logger, max_len=args.max_len)
    collate_fn = create_collate()

    return tokenizer, model, dataset, collate_fn


from posts_classifier.models.classifier import ClassifierTrainer as Trainer
from posts_classifier.models.classifier import ClassifierValidator as Validator
from posts_classifier.models.classifier import Tester


def classifier_fn(outputs, labels):
    print(outputs)
    raise Exception()


def main():
    tokenizer, model, dataset, collate_fn = prepare_posts_classification()

    # Because of some implementation stuff, we can't have __len__ on dataset
    dataset_size = len(dataset)  # type: ignore
    train_size = int(0.7 * dataset_size)
    eval_size = int(0.1 * dataset_size)
    test_size = dataset_size - (train_size + eval_size)
    train_dataset, eval_dataset, test_dataset = random_split(dataset, [train_size, eval_size, test_size])
    print("Performing training")
    validator = Validator(args=args, dataset=eval_dataset, collate_fn=collate_fn,)
    trainer = Trainer(model, tokenizer, args, train_dataset, logger, collate_fn, validator, eval_dataset)
    model = trainer.train()


if __name__ == "__main__":
    main()
