import torch
import utils
import metrics
import argparse
from dataset import AspectLinkDataset
from model import AspectLinkModel
from transformers import AutoTokenizer
from dataloader import AspectLinkDataLoader


def test(model, data_loader, run_file, qrels, metric, eval_run, device):
    res_dict = utils.evaluate7(
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
    parser.add_argument('--checkpoint', help='Name of checkpoint to load.', required=True, type=str)
    parser.add_argument('--save', help='Output run file in TREC format.', required=True,
                        type=str)
    parser.add_argument('--metric', help='Metric to use for evaluation. Default: map', default='map', type=str)
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

    cuda_device = 'cuda:' + str(args.cuda)
    print('CUDA Device: {} '.format(cuda_device))

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

    print('Creating test set...')
    test_set = AspectLinkDataset(
        dataset=args.test,
        tokenizer=tokenizer,
        train=False,
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

    model = AspectLinkModel(pretrained=pretrain)

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
