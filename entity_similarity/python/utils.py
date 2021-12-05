import configparser
import os
import sys
import time
import json
import torch
import torch.nn as nn
import warnings
import argparse
import time
import datetime
from typing import Tuple
from torch.utils.data import Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup


def format_time(elapsed):
    """
    Takes a time in seconds and returns a string hh:mm:ss
    """
    # Round to the nearest second.
    elapsed_rounded = int(round(elapsed))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def evaluate(model, loss_fn, data_loader, device):
    model.eval()
    epoch_loss = 0
    epoch_accuracy = 0

    with torch.no_grad():
        for batch in data_loader:
            e1_embedding = model(
                context_input_ids=batch['context_input_ids'].to(device),
                context_attention_mask=batch['context_attention_mask'].to(device),
                desc_input_ids=batch['e1_desc_input_ids'].to(device),
                desc_attention_mask=batch['e1_desc_attention_mask'].to(device),
                name_input_ids=batch['e1_name_input_ids'].to(device),
                name_attention_mask=batch['e1_name_attention_mask'].to(device),
                type_input_ids=batch['e1_type_input_ids'].to(device),
                type_attention_mask=batch['e1_type_attention_mask'].to(device)
            )
            e2_embedding = model(
                context_input_ids=batch['context_input_ids'].to(device),
                context_attention_mask=batch['context_attention_mask'].to(device),
                desc_input_ids=batch['e2_desc_input_ids'].to(device),
                desc_attention_mask=batch['e2_desc_attention_mask'].to(device),
                name_input_ids=batch['e2_name_input_ids'].to(device),
                name_attention_mask=batch['e2_name_attention_mask'].to(device),
                type_input_ids=batch['e2_type_input_ids'].to(device),
                type_attention_mask=batch['e2_type_attention_mask'].to(device)
            )
            similarity = torch.cosine_similarity(e1_embedding, e2_embedding).unsqueeze(dim=1)
            if isinstance(loss_fn, nn.BCEWithLogitsLoss):
                batch_loss = loss_fn(similarity, batch['label'].unsqueeze(dim=1).float().to(device))
            else:
                batch_loss = loss_fn(e1_embedding, e2_embedding, batch['label'].to(device))

            accuracy = binary_accuracy(similarity, batch['label'].to(self.device))

            epoch_loss += batch_loss.item()
            epoch_accuracy += accuracy.item()

    return epoch_loss / len(data_loader), epoch_accuracy / len(data_loader)


def binary_accuracy(predictions, y):
    rounded_predictions = torch.round(predictions)
    correct = (rounded_predictions == y).float()
    acc = correct.sum() / len(correct)
    return acc