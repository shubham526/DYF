import configparser
import os
import sys
import time
import json
import torch
import torch.nn as nn
import utils
import metrics
import warnings
import argparse
from typing import Tuple
from dataset import AspectLinkWithQuerySpecificEntityEmbeddingDataset, AspectLinkWithQueryIndependentEntityEmbeddingDataset
from torch.utils.data import Dataset
from model1 import AspectLinkModel
from transformers import AutoTokenizer
from dataloader import AspectLinkDataLoader


def test(model, data_loader, run_file, qrels, metric, eval_run, device):
    res_dict = utils.evaluate1(
        model=model,
        data_loader=data_loader,
        device=device
    )

    print('Writing run file...')
    utils.save_trec(run_file, res_dict)
    print('[Done].')

    if eval_run:
        test_metric = metrics.get_metric(qrels=qrels, run=run_file, metric=metric)
        print('{} = {:.4f}'.format(metric, test_metric))

def main():
    parser = argparse.ArgumentParser("Script to test a model.")
    parser.add_argument('--model-name', help='Name of model (bert|distill_bert|deberta|roberta). Default: bert.',
                        type=str,
                        default='bert')
    parser.add_argument('--model-type', help='Type of model (pairwise|pointwise). Default: pairwise.', type=str,
                        default='pairwise')
    parser.add_argument('--test', help='Training data.', required=True, type=str)
    parser.add_argument('--max-len', help='Maximum length for truncation/padding. Default: 512', default=512, type=int)
    parser.add_argument('--qrels', help='Ground truth file in TREC format.', type=str)
    parser.add_argument('--entity-emb', help='Entity embeddings file.', required=True, type=str)
    parser.add_argument('--entity-emb-dim', help='Dimension of entity embeddings Default: 100.', type=int, default=100)
    parser.add_argument('--entity-emb-type', help='Type of entity embeddings used (specific|independent)', type=str,
                        required=True)
    parser.add_argument('--checkpoint', help='Name of checkpoint to load.', required=True, type=str)
    parser.add_argument('--save', help='Output run file in TREC format.', required=True,
                        type=str)
    parser.add_argument('--metric', help='Metric to use for evaluation. Default: map', default='map', type=str)
    parser.add_argument('--score-method', help='Similarity method to use (linear|bilinear|cosine)', default='bilinear',
                        type=str)
    parser.add_argument('--batch-size', help='Size of each batch. Default: 8.', type=int, default=8)
    parser.add_argument('--num-workers', help='Number of workers to use for DataLoader. Default: 0', type=int,
                        default=0)
    parser.add_argument('--cuda', help='CUDA device number. Default: 0.', type=int, default=0)
    parser.add_argument('--use-cuda', help='Whether or not to use CUDA. Default: False.', action='store_true')
    parser.add_argument('--eval-run', help='Whether or not to evaluate run file. Default: False.', action='store_true')
    args = parser.parse_args()

    cuda_device = 'cuda:' + str(args.cuda)
    print('CUDA Device: {} '.format(cuda_device))

    device = torch.device(
        cuda_device if torch.cuda.is_available() and args.use_cuda else 'cpu'
    )

    if args.entity_emb_type == 'specific':
        if args.entity_emb is None:
            raise RuntimeError('Test entity embedding file missing.')
        else:
            print('Loading test entity embeddings...')
            test_entity_embeddings = torch.load(args.entity_emb, map_location=device)
            print('[Done].')

    else:
        if args.entity_emb is None:
            raise RuntimeError('Entity embedding file missing.')
        else:
            print('Loading entity embeddings...')
            with open(args.entity_emb, 'r') as f:
                test_entity_embeddings: Dict[str, List[float]] = json.load(f)
            print('[Done].')


    cuda_device = 'cuda:' + str(args.cuda)
    print('CUDA Device: {} '.format(cuda_device))

    device = torch.device(
        cuda_device if torch.cuda.is_available() and args.use_cuda else 'cpu'
    )

    if args.model_name is not None:
        if args.model_name == 'bert':
            print('Using BERT model.')
            pretrain = vocab = 'bert-base-uncased'
        elif args.model_name == 'deberta':
            print('Using DeBERTa model.')
            pretrain = vocab = 'microsoft/deberta-base'
        elif args.model_name == 'roberta':
            print('Using RoBERTa model.')
            pretrain = vocab = 'roberta-base'
        elif args.model_name == 'distill_bert':
            print('Using DistilBERT model.')
            pretrain = vocab = 'distilbert-base-uncased'
        else:
            raise ValueError('Wrong model name.')
    else:
        print('Model name not specified. Defaulting to BERT.')
        pretrain = vocab = 'bert-base-uncased'

    tokenizer = AutoTokenizer.from_pretrained(vocab)

    if args.entity_emb_type == 'specific':
        print('Creating train set...')
        test_set = AspectLinkWithQuerySpecificEntityEmbeddingDataset(
            dataset=args.test,
            tokenizer=tokenizer,
            train=False,
            entity_embeddings=test_entity_embeddings,
            max_len=args.max_len,
            data_type=args.model_type
        )
        print('[Done].')
    else:
        print('Creating test set...')
        test_set = AspectLinkWithQueryIndependentEntityEmbeddingDataset(
            dataset=args.test,
            tokenizer=tokenizer,
            train=False,
            entity_embeddings=test_entity_embeddings,
            max_len=args.max_len,
            data_type=args.model_type
        )
        print('[Done].')


    print('Creating data loaders...')
    print('Number of workers = ' + str(args.num_workers))
    print('Batch Size = ' + str(args.batch_size))

    test_loader = AspectLinkDataLoader(
        dataset=test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    print('[Done].')

    print('Model Type: ' + args.model_type)

    model = AspectLinkModel(pretrained=pretrain, entity_emb_dim=args.entity_emb_dim, score_method=args.score_method)

    print('Loading checkpoint...')
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print('[Done].')

    print('Using device: {}'.format(device))
    model.to(device)

    print("Starting to test...")
    test(
        model=model,
        data_loader=test_loader,
        run_file=args.save,
        eval_run=args.eval_run,
        qrels=args.qrels,
        metric=args.metric,
        device=device
    )

    print('Test complete.')


if __name__ == '__main__':
    main()
