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

sys.path.append("/home/sc1242/work/PyCharmProjects/bert_for_ranking/venv/src")


def train(model, trainer, epochs, valid_loader, save_path, save, patience, delta, device):
    train_stats_file = os.path.join(save_path, 'training_stats.json')
    if exists(train_stats_file):
        training_stats = json.load(open(train_stats_file))
        print('Loaded train stats file from ==> {}'.format(train_stats_file))
    else:
        training_stats = []

    total_t0 = time.time()
    best_score = None
    counter: int = 0
    early_stop: bool = False

    for epoch in range(epochs):
        print("")
        print('Training...')
        start_time = time.time()
        train_loss = trainer.train()
        print("Running Validation...")
        valid_loss = utils.evaluate(
            model=model,
            loss_fn=trainer.loss_fn,
            data_loader=valid_loader,
            device=device
        )
        training_time = utils.format_time(time.time() - start_time)

        score = -valid_loss
        if best_score is None:
            best_score = score
            torch.save(model.state_dict(), os.path.join(save_path, save))
            print('Model saved to ==> {}'.format(os.path.join(save_path, save)))
        elif score <= best_score + delta:
            counter += 1
            print(f'EarlyStopping counter: {counter} / {patience}')
            if counter >= patience:
                early_stop = True
        else:
            best_score = score
            torch.save(model.state_dict(), os.path.join(save_path, save))
            print('Model saved to ==> {}'.format(os.path.join(save_path, save)))
            counter = 0

        # Record all statistics from this epoch.
        training_stats.append({
            'Epoch': epoch + 1,
            'Training Loss': train_loss,
            'Validation Loss': valid_loss,
            'Training Time': training_time,
        })

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {training_time}')
        print(f'\tTrain Loss: {train_loss:.10f} | Validation Loss: {valid_loss:.10f}')

        if early_stop:
            break

    if early_stop:
        print('Early stopping patience reached. Training stopped.')


    with open(train_stats_file, 'w') as f:
        json.dump(training_stats, f)

    print("")
    print("Training Complete!")
    print('Train statistics saved to ==> {}'.format(train_stats_file))
    print("Total Training Time {:} (h:mm:ss)".format(utils.format_time(time.time() - total_t0)))


def main():
    parser = argparse.ArgumentParser("Script to train a model.")
    parser.add_argument('--bert-model', help='Name of model (bert|distill_bert|deberta|roberta). Default: bert.', type=str,
                        default='bert')
    parser.add_argument('--train', help='Training data.', required=True, type=str)
    parser.add_argument('--train-type', help='Type of training data (labelled|triplet).', required=True, type=str)
    parser.add_argument('--max-len', help='Maximum length for truncation/padding. Default: 512', default=512, type=int)
    parser.add_argument('--save-dir', help='Directory where model is saved.', required=True, type=str)
    parser.add_argument('--dev', help='Development data.', required=True, type=str)
    parser.add_argument('--save', help='Name of checkpoint to save. Default: model.bin', default='model.bin', type=str)
    parser.add_argument('--checkpoint', help='Name of checkpoint to load. Default: None', default=None, type=str)
    parser.add_argument('--epoch', help='Number of epochs. Default: 20', type=int, default=20)
    parser.add_argument('--loss-dist',
                        help='Distance function to use in loss function (euclidean|manhattan|cosine). Default: cosine',
                        type=str, default=None)
    parser.add_argument('--loss-margin',
                        help='Negative samples should have a distance of at least the margin value.'
                             'Default: 5 for TripletLoss and 0.5 for ContrastiveLoss', type=int, default=None)
    parser.add_argument('--loss-reduction',
                        help='Reduction to use in loss function (none|sum|mean).'
                             'Default: mean', type=str, default=None)
    parser.add_argument('--embed-dim', help='Dimension of embeddings. Default: 100', type=int, default=100)
    parser.add_argument('--ngram-sizes', help='Size of N-Grams. Default: [1,2,3]', nargs='+', type=int, default=[1, 2, 3])
    parser.add_argument('--batch-size', help='Size of each batch. Default: 8.', type=int, default=8)
    parser.add_argument('--learning-rate', help='Learning rate. Default: 2e-5.', type=float, default=2e-5)
    parser.add_argument('--n-warmup-steps', help='Number of warmup steps for scheduling. Default: 1000.', type=int,
                        default=1000)
    parser.add_argument('--num-workers', help='Number of workers to use for DataLoader. Default: 8', type=int,
                        default=8)
    parser.add_argument('--patience',
                        help='Number of epochs to wait after last validation loss improvement . Default: 8',
                        type=int, default=8)
    parser.add_argument('--delta',
                        help='Minimum change in the validation loss to qualify as an improvement. Default: 0.0001',
                        type=float, default=0.0001)
    parser.add_argument('--freeze-bert', help='Whether to freeze the BERT parameters for training. Default: False.',
                        action="store_true")
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

    model_config = json.dumps({
        'Max Input': args.max_len,
        'Model': pretrain,
        'Epochs': args.epoch,
        'Batch Size': args.batch_size,
        'Learning Rate': args.learning_rate,
        'Warmup Steps': args.n_warmup_steps,
    })
    config_file: str = os.path.join(args.save_dir, 'config.json')
    with open(config_file, 'w') as f:
        f.write("%s\n" % model_config)

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

    if args.freeze_bert:
        warnings.warn('BERT parameters frozen for training.')
        for param in model.context_encoder.embedding.bert.parameters():
            param.requires_grad = False

        for param in model.entity_encoder.desc_encoder.embedding.bert.parameters():
            param.requires_grad = False

        for param in model.entity_encoder.name_encoder.bert.parameters():
            param.requires_grad = False

        for param in model.entity_encoder.type_encoder.embedding.bert.parameters():
            param.requires_grad = False
    else:
        warnings.warn('BERT parameters not frozen for training. If you do want this, set the `--freeze-bert` flag.')

    dist = 'cosine' if args.loss_dist is None else args.loss_dist
    reduction = 'mean' if args.loss_reduction is None else args.loss_reduction

    if args.loss_margin is None:
        margin = 0.5 if args.train_type == 'labelled' else 5
    else:
        margin = args.loss_margin

    if args.train_type == 'labelled':
        print('Reading train data...')
        train_set = LabelledEntityPairDataset(
            dataset=args.train,
            tokenizer=tokenizer,
            max_len=512
        )
        print('[Done].')
        print('Reading dev data...')
        dev_set = LabelledEntityPairDataset(
            dataset=args.dev,
            tokenizer=tokenizer,
            max_len=512
        )
        print('[Done].')
        loss_fn = loss.ContrastiveLoss(
            distance_function=dist,
            margin=margin,
            reduction=reduction
        )
    elif args.train_type == 'triplet':
        print('Reading train data...')
        train_set = TripletDataset(
            dataset=args.train,
            tokenizer=tokenizer,
            max_len=512
        )
        print('[Done].')
        print('Reading dev data...')
        dev_set = TripletDataset(
            dataset=args.dev,
            tokenizer=tokenizer,
            max_len=512
        )
        print('[Done].')
        loss_fn = loss.TripletLoss(
            distance_function=dist,
            margin=margin,
            reduction=reduction
        )
    else:
        raise ValueError('ERROR! Train data type can be either `labelled` or `triplet`.')


    print('Creating data loaders...')
    print('Number of workers = ' + str(args.num_workers))
    print('Batch Size = ' + str(args.batch_size))
    train_loader = EntityDataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    dev_loader = EntityDataLoader(
        dataset=dev_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    print('[Done].')

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.n_warmup_steps,
        num_training_steps=len(train_set) * args.epoch // args.batch_size)

    print('Using device: {}'.format(device))
    model.to(device)
    loss_fn.to(device)

    if args.train_type == 'labelled':
        trainer = LabelledEntityPairTrainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            scheduler=scheduler,
            data_loader=train_loader,
            use_cuda=args.use_cuda,
            device=device
        )
    elif args.train_type == 'triplet':
        trainer = TripletTrainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            scheduler=scheduler,
            data_loader=train_loader,
            use_cuda=args.use_cuda,
            device=device
        )
    else:
        raise ValueError('ERROR! Train data type can be either `labelled` or `triplet`.')

    train(
        model=model,
        trainer=trainer,
        epochs=args.epoch,
        valid_loader=dev_loader,
        save_path=args.save_dir,
        save=args.save,
        patience=args.patience,
        delta=args.delta,
        device=device
    )


if __name__ == '__main__':
    main()
