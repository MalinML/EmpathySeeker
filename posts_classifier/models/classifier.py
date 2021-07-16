from typing import Tuple, Union
from transformers import BertTokenizer, BertForSequenceClassification


def get_all_labels_classifier(model_name: str) -> Tuple[BertForSequenceClassification, BertTokenizer]:
    tokenizer: BertTokenizer = BertTokenizer.from_pretrained(model_name)
    model: BertForSequenceClassification = BertForSequenceClassification.from_pretrained(model_name, num_labels=27)
    return model, tokenizer
