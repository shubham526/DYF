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
from os.path import exists
from typing import Tuple
from dataset import LabelledEntityPairDataset, TripletDataset
from dataloader import EntityDataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from model import ContextualEntityEmbedding
from abc import ABC, abstractmethod
from trainers import LabelledEntityPairTrainer, TripletTrainer


def get_data(data_file, entity_type):
    with open(data_file, 'rt') as f:
        for line in f:
            line_dict = json.loads(line)
            context = line_dict['context']
            entities = line_dict['context_entities'] if entity_type == 'context' else line_dict['aspect_entities']
            yield context, entities


def create_bert_input(text, tokenizer) -> Tuple[torch.Tensor, torch.Tensor]:
        # Tokenize all of the sentences and map the tokens to their word IDs.
        encoded_dict = tokenizer.encode_plus(
            text=text,
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=512,  # Pad & truncate all sentences.
            padding='max_length',
            truncation=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt'
        )

        return encoded_dict['input_ids'], encoded_dict['attention_mask']


def create_embeddings(model, data_file, entity_type, tokenizer, save, device):
    save_dict = {}
    with torch.no_grad():
        for context, entities in tqdm.tqdm(get_data(data_file=data_file, entity_type=entity_type)):
            context_input_ids, context_attention_mask = create_bert_input(context['text'], tokenizer)
            entity_dict = {}
            for entity in entities:
                if entity is not None:
                    desc_input_ids, desc_attention_mask = create_bert_input(entity['entity_desc'], tokenizer)
                    name_input_ids, name_attention_mask = create_bert_input(entity['entity_name'], tokenizer)
                    type_input_ids, type_attention_mask = create_bert_input(' '.join(entity['entity_types']), tokenizer)
                    embedding = model(
                        context_input_ids=context_input_ids.to(device),
                        context_attention_mask=context_attention_mask.to(device),
                        desc_input_ids=desc_input_ids.to(device),
                        desc_attention_mask=desc_attention_mask.to(device),
                        name_input_ids=name_input_ids.to(device),
                        name_attention_mask=name_attention_mask.to(device),
                        type_input_ids=type_input_ids.to(device),
                        type_attention_mask=type_attention_mask.to(device)
                    )
                    entity_dict[entity['entity_id']] = embedding
                save_dict[context['id']] = entity_dict


        print('Saving embeddings...')
        torch.save(save_dict, save)
        print('[Done].')




def main():
    parser = argparse.ArgumentParser("Script to create embeddings.")
    parser.add_argument('--bert-model', help='Name of BERT model (bert|distill_bert|deberta|roberta). Default: bert.',
                        type=str, default='bert')
    parser.add_argument('--data', help='Data file.', required=True, type=str)
    parser.add_argument('--entity-type', help='Type of entity (context|aspect).', required=True, type=str)
    parser.add_argument('--save', help='File to save.', required=True, type=str)
    parser.add_argument('--checkpoint', help='Name of checkpoint to load. Default: None', default=None, type=str)
    parser.add_argument('--embed-dim', help='Dimension of embeddings. Default: 100', type=int, default=100)
    parser.add_argument('--ngram-sizes', help='Size of N-Grams. Default: [1,2,3]', nargs='+', type=int,
                        default=[1, 2, 3])
    parser.add_argument('--cuda', help='CUDA device number. Default: 0.', type=int, default=0)
    parser.add_argument('--use-cuda', help='Whether or not to use CUDA. Default: False.', action='store_true')
    args = parser.parse_args()

    cuda_device = 'cuda:' + str(args.cuda)
    print('CUDA Device: {} '.format(cuda_device))

    device = torch.device(
        cuda_device if torch.cuda.is_available() and args.use_cuda else 'cpu'
    )

    if args.bert_model is not None:
        if args.bert_model == 'bert':
            print('Using BERT model.')
            pretrain = vocab = 'bert-base-uncased'
        elif args.bert_model == 'deberta':
            print('Using DeBERTa model.')
            pretrain = vocab = 'microsoft/deberta-base'
        elif args.bert_model == 'roberta':
            print('Using RoBERTa model.')
            pretrain = vocab = 'roberta-base'
        elif args.bert_model == 'distill_bert':
            print('Using DistilBERT model.')
            pretrain = vocab = 'distilbert-base-uncased'
        else:
            raise ValueError('Wrong model name.')
    else:
        print('Model name not specified. Defaulting to BERT.')
        pretrain = vocab = 'bert-base-uncased'

    tokenizer = AutoTokenizer.from_pretrained(vocab)
    model = ContextualEntityEmbedding(
        pretrained_bert_model=pretrain,
        out_dim=args.embed_dim,
        n_gram_sizes=args.ngram_sizes
    )
    if args.checkpoint is not None:
        print('Loading checkpoint...')
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print('[Done].')

    print('Using device: {}'.format(device))
    model.eval()
    model.to(device)

    create_embeddings(model=model, data_file=args.data, entity_type=args.entity_type, tokenizer=tokenizer, save=args.save, device=device)



if __name__ == '__main__':
    main()



