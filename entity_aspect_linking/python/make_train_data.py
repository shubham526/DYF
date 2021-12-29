import json
import gzip
import sys
from utils import tqdm_joblib
from tqdm import tqdm
import os
import argparse
import utils
from joblib import Parallel, delayed
from typing import List, Dict, Set, Tuple, Any
from pykson import Pykson, JsonObject, StringField, IntegerField, ListField, ObjectListField, ObjectField, Pykson, \
    BooleanField
from object_models import Location, Entity, AnnotatedText, AspectLinkExample, Aspect, Context

totals = {
    'nanni-test.jsonl.gz': 18289,
    'overly-frequent.jsonl.gz': 429160,
    'test.jsonl.gz': 4967,
    'train-remaining.jsonl.gz': 544892,
    'train-small.jsonl.gz': 5498,
    'validation.jsonl.gz': 4313

}
processor = utils.TextProcessor()

def to_pairwise_data(example: AspectLinkExample, context_type: str) -> List[str]:
    data: List[str] = []
    query: Dict[str, Any] = {
        'text': processor.preprocess(example.context.sentence.content) if context_type == 'sent' else processor.preprocess(example.context.paragraph.content),
        'entities': utils.get_entity_ids_only(example.context.sentence.entities) if context_type == 'sent' else utils.get_entity_ids_only(example.context.paragraph.entities)
    }

    # query: str = processor.preprocess(example.context.sentence.content) if context_type == 'sent' else processor.preprocess(example.context.paragraph.content)
    query_id: str = example.id
    doc_pos: Dict[str, Any] = utils.get_positive_doc(example.candidate_aspects, example.true_aspect)
    doc_neg_list: List[Tuple[str, Dict[str, Any]]] = utils.get_negative_doc_list(example.candidate_aspects, example.true_aspect)
    documents: List[Tuple[Dict[str, Any], Dict[str, Any]]] = [
        (doc_pos, doc_neg) for _, doc_neg in doc_neg_list
    ]

    for doc_pos, doc_neg in documents:
        data.append(
            json.dumps({
                'query': query,
                'query_id': query_id,
                'doc_pos': doc_pos,
                'doc_neg': doc_neg
            })
        )
    return data


def to_pointwise_data(example: AspectLinkExample, context_type: str) -> List[str]:
    data: List[str] = []
    query: Dict[str, Any] = {
        'text': processor.preprocess(
            example.context.sentence.content) if context_type == 'sent' else processor.preprocess(
            example.context.paragraph.content),
        'entities': utils.get_entity_ids_only(
            example.context.sentence.entities) if context_type == 'sent' else utils.get_entity_ids_only(
            example.context.paragraph.entities)
    }
    # query: str = processor.preprocess(example.context.sentence.content) if context_type == 'sent' else processor.preprocess(example.context.paragraph.content)
    query_id: str = example.id
    doc_pos: Dict[str, Any] = utils.get_positive_doc(example.candidate_aspects, example.true_aspect)
    doc_neg_list: List[Tuple[str, Dict[str, Any]]] = utils.get_negative_doc_list(example.candidate_aspects, example.true_aspect)
    data.append(json.dumps({
        'query': query,
        'query_id': query_id,
        'doc': doc_pos,
        'label': 1
    }))
    for _, doc_neg in doc_neg_list:
        data.append(json.dumps({
            'query': query,
            'query_id': query_id,
            'doc': doc_neg,
            'label': 0
        }))
    return data



def create_data(data_type: str, data_file: str, save: str, context_type: str, num_workers: int) -> None:
    print('Data type: {}'.format(data_type))
    print('Context type: {}'.format(context_type))
    print('Number of processes = {}'.format(num_workers))
    total = totals[os.path.basename(data_file)]

    if data_type == 'pairwise':
        with tqdm_joblib(tqdm(desc="Progress", total=total)) as progress_bar:
            data = Parallel(n_jobs=num_workers, backend='multiprocessing')(
                delayed(to_pairwise_data)(example, context_type) for example in utils.aspect_link_examples(data_file))
    elif data_type == 'pointwise':
        with tqdm_joblib(tqdm(desc="Progress", total=total)) as progress_bar:
            data = Parallel(n_jobs=num_workers, backend='multiprocessing')(
                delayed(to_pointwise_data)(example, context_type) for example in utils.aspect_link_examples(data_file))
    else:
        raise ValueError('Mode must be `pairwise` or `pointwise`.')

    print('Writing to file...')
    for d in data:
        write_to_file(d, save)
    print('[Done].')
    print('File written to ==> {}'.format(save))


def write_to_file(data: List[str], output_file: str):
    with open(output_file, 'a') as f:
        for line in data:
            f.write("%s\n" % line)


def main():
    parser = argparse.ArgumentParser("Create a training file.")
    parser.add_argument("--mode", help="Type of data (pairwise|pointwise).", required=True, type=str)
    parser.add_argument("--data", help="Data file.", required=True, type=str)
    parser.add_argument("--save", help="Output file.", required=True, type=str)
    parser.add_argument("--context", help="Type of context to use (sent|para). Default: paragraph context.",
                        default='para', type=str)
    parser.add_argument("--num-workers", help="Number of processes to use. Default: 4.",
                        default=4, type=int)
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
    create_data(args.mode, args.data, args.save, args.context, args.num_workers)


if __name__ == '__main__':
    main()

