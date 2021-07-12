import os

import numpy as np
import torch
from coai_base.training.trainer import Trainer
from coai_base.training.validator import Validator
from coai_base.utils import save_checkpoint
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm.auto import tqdm, trange
from transformers import AdamW, BertModel, get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter


class Bert(torch.nn.Module):
    def __init__(self) -> None:
        super(Bert, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased", cache_dir="cached")
        self.dropout = torch.nn.Dropout(0.2)
        self.linear = torch.nn.Linear(768, 8)

    def forward(self, input_ids):
        outputs = self.bert(input_ids)
        outputs = self.dropout(outputs.pooler_output)
        outputs = self.linear(outputs)
        return outputs


class ClassifierValidator(Validator):
    def _batch_to_model_input(self, batch):
        input_ids, _ = batch
        input_ids = input_ids.to(self.args.device)
        return {"input_ids": input_ids}

    def _get_labels_from_batch(self, batch):
        _, labels = batch
        return labels


class ClassifierTrainer(Trainer):
    def _batch_to_model_input(self, batch):
        input_ids, _ = batch
        input_ids = input_ids.to(self.args.device)
        return {"input_ids": input_ids}

    def _get_labels_from_batch(self, batch):
        _, labels = batch
        return labels


class Tester:
    def __init__(self, args, tokenizer, dataset, collate_fn, logger):
        self.args = args
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.logger = logger

        self.dataloader = self.create_loader()

    def create_loader(self):
        return DataLoader(self.dataset, drop_last=False, collate_fn=self.collate_fn, batch_size=self.args.batch_size, shuffle=True,)

    def test(self, model):
        print(">>> Testing")

        # Iterator
        data_iterator = tqdm(self.dataloader, desc="Test")
        targets = []
        predictions = []
        correct_preds = 0
        total_samples = 0
        # Evaluation
        for _, batch in enumerate(data_iterator):
            input_ids, labels = batch
            input_ids = input_ids.to(self.args.device)
            outputs = model(input_ids)
            preds = torch.max(torch.sigmoid(outputs).detach().cpu(), dim=1).indices.numpy()
            labels = torch.max(labels, dim=1).indices.numpy()

            correct_preds += np.sum(preds == labels)

            targets.extend(labels)
            predictions.extend(preds)

            num_samples = input_ids.shape[0]
            total_samples += num_samples
            targets.extend(labels)
            predictions.extend(preds)

        precision, recall, f1, _ = precision_recall_fscore_support(targets, predictions)
        acc = accuracy_score(targets, predictions)
        print(f"Accuracy: {correct_preds}, {total_samples}, {acc}%")
        print(f"{precision} and {recall} and {f1}")

        print(">>> End of Generation <<<")
