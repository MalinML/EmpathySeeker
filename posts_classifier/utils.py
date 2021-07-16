import random
import numpy as np
import torch
import logging


def set_seed(seed: int = 42, n_gpu: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def get_logger(file_name: str, logger_name: str = "dialogue") -> logging.Logger:
    root = logging.getLogger(logger_name)
    root.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s | %(message)s")
    file_handler = logging.FileHandler(file_name)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)
    return root
