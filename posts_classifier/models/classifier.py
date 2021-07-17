from typing import Optional, Tuple, Union
from transformers import BertTokenizer, BertForSequenceClassification


def get_all_labels_classifier(model_name: str, model_override_name: Optional[str]) -> Tuple[BertForSequenceClassification, BertTokenizer]:
    print("Getting all labels classifier")
    model_override_name = model_override_name if model_override_name is not None else model_name
    tokenizer: BertTokenizer = BertTokenizer.from_pretrained(model_name)
    model: BertForSequenceClassification = BertForSequenceClassification.from_pretrained(model_override_name, num_labels=27)
    return model, tokenizer
