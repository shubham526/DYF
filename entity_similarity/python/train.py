import configparser
import os
import sys
import time
import json
import torch
import torch.nn as nn
import utils
import warnings
import argparse
import datetime
from typing import Tuple
from dataset import EntitySimilarityDataset
from dataloader import EntitySimilarityDataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from model import ContextualEntityEmbedding

sys.path.append("/home/sc1242/work/PyCharmProjects/bert_for_ranking/venv/src")

class Trainer:
    def __init__(self, model, optimizer, loss_fn, scheduler, data_loader, use_cuda, device):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.data_loader = data_loader
        self.use_cuda = use_cuda
        self.device = device

    def train(self, start_time):
        train_step = self.make_train_step()
        epoch_loss = 0
        epoch_accuracy = 0
        num_batch = len(self.data_loader)
        for step, batch in enumerate(self.data_loader):
            if step % 20 == 0 and not step == 0:
                elapsed = utils.format_time(time.time() - start_time)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, num_batch, elapsed))

            batch_loss, batch_accuracy = train_step(batch)
            epoch_loss += batch_loss
            epoch_accuracy += batch_accuracy

        return epoch_loss/num_batch, epoch_accuracy/num_batch

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

            similarity = torch.cosine_similarity(e1_embedding, e2_embedding).unsqueeze(dim=1)
            if isinstance(self.loss_fn, nn.BCEWithLogitsLoss):
                batch_loss = self.loss_fn(similarity, batch['label'].unsqueeze(dim=1).float().to(self.device))
            else:
                batch_loss = self.loss_fn(e1_embedding, e2_embedding, batch['label'].to(self.device))

            train_accuracy = utils.binary_accuracy(similarity, batch['label'].to(self.device))

            # Computes gradients
            batch_loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            # Updates parameters
            self.optimizer.step()
            self.scheduler.step()

            # Returns the loss
            return batch_loss.item(), train_accuracy.item()

        # Returns the function that will be called inside the train loop
        return train_step


def train(model, trainer, epochs, valid_loader, save_path, save, device):
    best_valid_acc = 0.0
    training_stats = []
    total_t0 = time.time()

    for epoch in range(epochs):
        # ========================================
        #               Training
        # ========================================

        # Perform one full pass over the training set.

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch + 1, epochs))
        print('Training...')
        start_time = time.time()
        train_loss, train_accuracy = trainer.train(start_time=start_time)
        training_time = utils.format_time(time.time() - start_time)

        print("")
        print("  Average Training Loss: {0:.2f}".format(train_loss))
        print("  Average Training Accuracy: {0:.2f}".format(train_accuracy))
        print("  Train Time: {:}".format(training_time))

        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        print("")
        print("Running Validation...")
        start_time = time.time()
        valid_loss, valid_accuracy = utils.evaluate(
            model=model,
            loss_fn=trainer.loss_fn,
            data_loader=valid_loader,
            device=device
        )
        validation_time = utils.format_time(time.time() - start_time)
        if valid_accuracy >= best_valid_acc:
            best_valid_acc = valid_accuracy
            # Save the checkpoint
            torch.save(model.state_dict(), os.path.join(save_path, save))
            print(f'Model saved to ==> {os.path.join(save_path, save)}')

        print("  Average Validation Loss: {0:.2f}".format(valid_loss))
        print("  Average Validation Accuracy: {0:.2f}".format(valid_accuracy))
        print("  Validation Time: {:}".format(validation_time))

        # Record all statistics from this epoch.
        training_stats.append({
            'Epoch': epoch + 1,
            'Training Loss': train_loss,
            'Validation Loss': valid_loss,
            'Validation Accuracy.': valid_accuracy,
            'Training Time': training_time,
            'Validation Time': validation_time
        })

    train_stats_file = os.path.join(save_path, "training_stats.json")
    with open(train_stats_file, 'w') as f:
        json.dump(training_stats, f)

    print("")
    print("Training Complete!")
    print('Train statistics saved to ==> {}'.format(tran_stats_file))
    print("Total Training Time {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))


def main():
    parser = argparse.ArgumentParser("Script to train a model.")
    parser.add_argument('--bert-model', help='Name of model (bert|distill_bert|deberta|roberta). Default: bert.', type=str,
                        default='bert')
    parser.add_argument('--train', help='Training data.', required=True, type=str)
    parser.add_argument('--max-len', help='Maximum length for truncation/padding. Default: 512', default=512, type=int)
    parser.add_argument('--save-dir', help='Directory where model is saved.', required=True, type=str)
    parser.add_argument('--dev', help='Development data.', required=True, type=str)
    parser.add_argument('--save', help='Name of checkpoint to save. Default: model.bin', default='model.bin', type=str)
    parser.add_argument('--checkpoint', help='Name of checkpoint to load. Default: None', default=None, type=str)
    parser.add_argument('--epoch', help='Number of epochs. Default: 20', type=int, default=20)
    parser.add_argument('--loss-fn', help='Name of loss function to use (binary_cross_entropy|cosine_embedding). '
                                          'Default: Binary Cross Entropy', type=str, default='binary_cross_entropy')
    parser.add_argument('--embed-dim', help='Dimension of embeddings. Default: 100', type=int, default=100)
    parser.add_argument('--ngram-sizes', help='Size of N-Grams. Default: [1,2,3]', nargs='+', type=int, default=[1, 2, 3])
    parser.add_argument('--batch-size', help='Size of each batch. Default: 8.', type=int, default=8)
    parser.add_argument('--learning-rate', help='Learning rate. Default: 2e-5.', type=float, default=2e-5)
    parser.add_argument('--n-warmup-steps', help='Number of warmup steps for scheduling. Default: 1000.', type=int,
                        default=1000)
    parser.add_argument('--num-workers', help='Number of workers to use for DataLoader. Default: 8', type=int,
                        default=8)
    parser.add_argument('--use-cuda', help='Whether or not to use CUDA. Default: False.', action='store_true')
    args = parser.parse_args()

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
    print('Reading train data...')
    train_set = EntitySimilarityDataset(
        dataset=args.train,
        tokenizer=tokenizer,
        max_len=512
    )
    print('[Done].')
    print('Reading dev data...')
    dev_set = EntitySimilarityDataset(
        dataset=args.dev,
        tokenizer=tokenizer,
        max_len=512
    )
    print('[Done].')

    print('Creating data loaders...')
    print('Number of workers = ' + str(args.num_workers))
    print('Batch Size = ' + str(args.batch_size))
    train_loader = EntitySimilarityDataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    dev_loader = EntitySimilarityDataLoader(
        dataset=dev_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    print('[Done].')

    model = ContextualEntityEmbedding(pretrained_bert_model=pretrain, out_dim=args.embed_dim, n_gram_sizes=args.ngram_sizes)
    if args.loss_fn == 'binary_cross_entropy':
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        loss_fn = nn.CosineEmbeddingLoss()

    if args.checkpoint is not None:
        print('Loading checkpoint...')
        model.load_state_dict(torch.load(args.checkpoint))
        print('[Done].')

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.n_warmup_steps,
        num_training_steps=len(train_set) * args.epoch // args.batch_size)

    device = torch.device(
        'cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu'
    )
    print('Using device: {}'.format(device))
    model.to(device)
    loss_fn.to(device)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        scheduler=scheduler,
        data_loader=train_loader,
        use_cuda=args.use_cuda,
        device=device
    )

    train(
        model=model,
        trainer=trainer,
        epochs=args.epoch,
        valid_loader=dev_loader,
        save_path=args.save_dir,
        save=args.save,
        device=device
    )


if __name__ == '__main__':
    main()
