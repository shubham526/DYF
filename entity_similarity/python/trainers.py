import configparser
import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import loss
import tqdm
import utils
import warnings
import argparse
import datetime
from typing import Tuple
from dataset import LabelledEntityPairDataset
from dataloader import EntityDataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from model import ContextualEntityEmbedding
from abc import ABC, abstractmethod

class Trainer(ABC):
    def __init__(self, model, optimizer, loss_fn, scheduler, data_loader, use_cuda, device):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.data_loader = data_loader
        self.use_cuda = use_cuda
        self.device = device

    def train(self):
        train_step = self.make_train_step()
        epoch_loss = 0
        epoch_accuracy = 0
        num_batch = len(self.data_loader)
        for _, batch in tqdm.tqdm(enumerate(self.data_loader), total=num_batch):
            batch_loss = train_step(batch)
            epoch_loss += batch_loss
            # epoch_accuracy += batch_accuracy

        return epoch_loss/num_batch

    @abstractmethod
    def make_train_step(self):
        # Builds function that performs a step in the train loop
        pass


class LabelledEntityPairTrainer(Trainer):

    def __init__(self, model, optimizer, loss_fn, scheduler, data_loader, use_cuda, device):
        super(LabelledEntityPairTrainer, self).__init__(model, optimizer, loss_fn, scheduler, data_loader, use_cuda, device)

    def make_train_step(self):
        # Builds function that performs a step in the train loop
        def train_step(batch):
            # Sets model to TRAIN mode
            self.model.train()

            # Zero the gradients
            self.optimizer.zero_grad()

            # Create embeddings nd compute loss
            e1_embedding = self.model(
                context_input_ids=batch['context_input_ids'].to(self.device),
                context_attention_mask=batch['context_attention_mask'].to(self.device),
                desc_input_ids=batch['e1_desc_input_ids'].to(self.device),
                desc_attention_mask=batch['e1_desc_attention_mask'].to(self.device),
                name_input_ids=batch['e1_name_input_ids'].to(self.device),
                name_attention_mask=batch['e1_name_attention_mask'].to(self.device),
                type_input_ids=batch['e1_type_input_ids'].to(self.device),
                type_attention_mask=batch['e1_type_attention_mask'].to(self.device)
            )
            e2_embedding = self.model(
                context_input_ids=batch['context_input_ids'].to(self.device),
                context_attention_mask=batch['context_attention_mask'].to(self.device),
                desc_input_ids=batch['e2_desc_input_ids'].to(self.device),
                desc_attention_mask=batch['e2_desc_attention_mask'].to(self.device),
                name_input_ids=batch['e2_name_input_ids'].to(self.device),
                name_attention_mask=batch['e2_name_attention_mask'].to(self.device),
                type_input_ids=batch['e2_type_input_ids'].to(self.device),
                type_attention_mask=batch['e2_type_attention_mask'].to(self.device)
            )

            batch_loss = self.loss_fn(e1_embedding, e2_embedding, batch['label'].to(self.device))
            # distances = self.loss_fn.distance_function(e1_embedding, e2_embedding).unsqueeze(dim=0)
            # predictions = F.softmax(distances, dim=1)
            # batch_accuracy = utils.binary_accuracy(predictions, batch['label'].to(self.device))


            # Computes gradients
            batch_loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            #torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            # Updates parameters
            self.optimizer.step()
            self.scheduler.step()

            # Returns the loss
            return batch_loss.item()

        # Returns the function that will be called inside the train loop
        return train_step


class TripletTrainer(Trainer):
    def __init__(self, model, optimizer, loss_fn, scheduler, data_loader, use_cuda, device):
        super(TripletTrainer, self).__init__(model, optimizer, loss_fn, scheduler, data_loader, use_cuda, device)

    def make_train_step(self):
        # Builds function that performs a step in the train loop
        def train_step(batch):
            # Sets model to TRAIN mode
            self.model.train()

            # Zero the gradients
            self.optimizer.zero_grad()

            # Create embeddings and compute loss
            anchor_embedding = self.model(
                context_input_ids=batch['context_input_ids'].to(self.device),
                context_attention_mask=batch['context_attention_mask'].to(self.device),
                desc_input_ids=batch['anchor_desc_input_ids'].to(self.device),
                desc_attention_mask=batch['anchor_desc_attention_mask'].to(self.device),
                name_input_ids=batch['anchor_name_input_ids'].to(self.device),
                name_attention_mask=batch['anchor_name_attention_mask'].to(self.device),
                type_input_ids=batch['anchor_type_input_ids'].to(self.device),
                type_attention_mask=batch['anchor_type_attention_mask'].to(self.device)
            )
            pos_embedding = self.model(
                context_input_ids=batch['context_input_ids'].to(self.device),
                context_attention_mask=batch['context_attention_mask'].to(self.device),
                desc_input_ids=batch['pos_desc_input_ids'].to(self.device),
                desc_attention_mask=batch['pos_desc_attention_mask'].to(self.device),
                name_input_ids=batch['pos_name_input_ids'].to(self.device),
                name_attention_mask=batch['pos_name_attention_mask'].to(self.device),
                type_input_ids=batch['pos_type_input_ids'].to(self.device),
                type_attention_mask=batch['pos_type_attention_mask'].to(self.device)
            )
            neg_embedding = self.model(
                context_input_ids=batch['context_input_ids'].to(self.device),
                context_attention_mask=batch['context_attention_mask'].to(self.device),
                desc_input_ids=batch['neg_desc_input_ids'].to(self.device),
                desc_attention_mask=batch['neg_desc_attention_mask'].to(self.device),
                name_input_ids=batch['neg_name_input_ids'].to(self.device),
                name_attention_mask=batch['neg_name_attention_mask'].to(self.device),
                type_input_ids=batch['neg_type_input_ids'].to(self.device),
                type_attention_mask=batch['neg_type_attention_mask'].to(self.device)
            )


            batch_loss = self.loss_fn(anchor_embedding, pos_embedding, neg_embedding)

            # Computes gradients
            batch_loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            #torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            # Updates parameters
            self.optimizer.step()
            self.scheduler.step()

            # Returns the loss
            return batch_loss.item()

        # Returns the function that will be called inside the train loop
        return train_step

