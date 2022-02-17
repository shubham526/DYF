import json
import gzip
import sys
from utils import tqdm_joblib
from tqdm import tqdm
import os
import argparse
from joblib import Parallel, delayed
import utils
from typing import List, Dict, Set, Tuple, Any
from pykson import Pykson, JsonObject, StringField, IntegerField, ListField, ObjectListField, ObjectField, Pykson, \
    BooleanField
from object_models import Location, Entity, AnnotatedText, AspectLinkExample, Aspect, Context

processor = utils.TextProcessor()

totals = {
    'nanni-test.jsonl.gz': 18289,
    'overly-frequent.jsonl.gz': 429160,
    'test.jsonl.gz': 4967,
    'train-remaining.jsonl.gz': 544892,
    'train-small.jsonl.gz': 5498,
    'validation.jsonl.gz': 4313

}


def to_data(example: AspectLinkExample, context_type: str) -> List[str]:
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
    doc_pos: str = utils.get_positive_doc(example.candidate_aspects, example.true_aspect)
    doc_pos_id: str = example.true_aspect
    doc_neg_list: List[Tuple[str, Dict[str, Any]]] = utils.get_negative_doc_list(example.candidate_aspects, example.true_aspect)

    if doc_pos and len(doc_neg_list) >= 1 and len(query['entities']) != 0:
        data.append(json.dumps({
            'query': query,
            'doc': doc_pos,
            'label': 1,
            'query_id': query_id,
            'doc_id': doc_pos_id
        }))

        for doc_neg_id, doc_neg in doc_neg_list:
            data.append(json.dumps({
                'query': query,
                'doc': doc_neg,
                'label': 0,
                'query_id': query_id,
                'doc_id': doc_neg_id
            }))



    return data


def create_data(data_file: str, save: str, context_type: str, num_workers: int) -> None:
    print('Context type: {}'.format(context_type))
    print('Number of processes = {}'.format(num_workers))
    total = totals[os.path.basename(data_file)]
    with tqdm_joblib(tqdm(desc="Progress", total=total)) as progress_bar:
        data = Parallel(n_jobs=num_workers, backend='multiprocessing')(
            delayed(to_data)(example, context_type) for example in utils.aspect_link_examples(data_file))

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
    parser = argparse.ArgumentParser("Create a validation or test file.")
    parser.add_argument("--data", help="Data file.", required=True)
    parser.add_argument("--save", help="Output file.", required=True)
    parser.add_argument("--context", help="Type of context to use (sent|para). Default: paragraph context.",
                        default='para')
    parser.add_argument("--num-workers", help="Number of processes to use. Default: 4.",
                        default=4, type=int)
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
    create_data( args.data, args.save, args.context, args.num_workers)


if __name__ == '__main__':
    main()
