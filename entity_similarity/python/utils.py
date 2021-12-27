import configparser
import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import argparse
import time
import loss
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
            if isinstance(loss_fn, loss.ContrastiveLoss):
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
                batch_loss = loss_fn(e1_embedding, e2_embedding, batch['label'].to(device))
            else:
                anchor_embedding = model(
                    context_input_ids=batch['context_input_ids'].to(device),
                    context_attention_mask=batch['context_attention_mask'].to(device),
                    desc_input_ids=batch['anchor_desc_input_ids'].to(device),
                    desc_attention_mask=batch['anchor_desc_attention_mask'].to(device),
                    name_input_ids=batch['anchor_name_input_ids'].to(device),
                    name_attention_mask=batch['anchor_name_attention_mask'].to(device),
                    type_input_ids=batch['anchor_type_input_ids'].to(device),
                    type_attention_mask=batch['anchor_type_attention_mask'].to(device)
                )
                pos_embedding = model(
                    context_input_ids=batch['context_input_ids'].to(device),
                    context_attention_mask=batch['context_attention_mask'].to(device),
                    desc_input_ids=batch['pos_desc_input_ids'].to(device),
                    desc_attention_mask=batch['pos_desc_attention_mask'].to(device),
                    name_input_ids=batch['pos_name_input_ids'].to(device),
                    name_attention_mask=batch['pos_name_attention_mask'].to(device),
                    type_input_ids=batch['pos_type_input_ids'].to(device),
                    type_attention_mask=batch['pos_type_attention_mask'].to(device)
                )
                neg_embedding = model(
                    context_input_ids=batch['context_input_ids'].to(device),
                    context_attention_mask=batch['context_attention_mask'].to(device),
                    desc_input_ids=batch['neg_desc_input_ids'].to(device),
                    desc_attention_mask=batch['neg_desc_attention_mask'].to(device),
                    name_input_ids=batch['neg_name_input_ids'].to(device),
                    name_attention_mask=batch['neg_name_attention_mask'].to(device),
                    type_input_ids=batch['neg_type_input_ids'].to(device),
                    type_attention_mask=batch['neg_type_attention_mask'].to(device)
                )

                batch_loss = loss_fn(anchor_embedding, pos_embedding, neg_embedding)

            # distances = loss_fn.distance_function(e1_embedding, e2_embedding).unsqueeze(dim=0)
            # predictions = F.softmax(distances, dim=1)
            # batch_accuracy = binary_accuracy(predictions, batch['label'].to(device))
            epoch_loss += batch_loss.item()
            # epoch_accuracy += batch_accuracy.item()

    return epoch_loss / len(data_loader)


def binary_accuracy(predictions, y):
    rounded_predictions = torch.round(predictions)
    correct = (rounded_predictions == y).float()
    acc = correct.sum() / len(correct)
    return acc