import json
import gzip
import sys
import tqdm
import os
import argparse
import utils
from typing import List, Dict, Set, Tuple
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


def to_data(query: Tuple[str, str], doc_pos: Tuple[str, str], doc_neg_list: List[Tuple[str, str]]) -> List[str]:
    data: List[str] = [to_json(
        query=query[0],
        query_id=query[1],
        doc=doc_pos[0],
        doc_id=doc_pos[1],
        label='1'
    )]
    for doc_neg_id, doc_neg in doc_neg_list:
        data.append(to_json(
            query=query[0],
            query_id=query[1],
            doc=doc_neg,
            doc_id=doc_neg_id,
            label='0'
        ))

    return data


def to_json(query: str, doc: str, label: str, query_id: str, doc_id: str) -> str:
    return json.dumps({
        'query': query,
        'doc': doc,
        'label': label,
        'query_id': query_id,
        'doc_id': doc_id
    })



def create_data(data_file: str, save: str, context_type: str) -> None:
    total = totals[os.path.basename(data_file)]
    for example in tqdm.tqdm(utils.aspect_link_examples(data_file), total=total):
        query: str = processor.preprocess(example.context.sentence.content) if context_type == 'sent' else processor.preprocess(example.context.paragraph.content)
        query_id: str = example.id
        doc_pos: str = utils.get_positive_doc(example.candidate_aspects, example.true_aspect)
        doc_pos_id: str = example.true_aspect
        doc_neg_list: List[Tuple[str, str]] = utils.get_negative_doc_list(example.candidate_aspects, example.true_aspect)
        data: List[str] = to_data((query, query_id), (doc_pos, doc_pos_id), doc_neg_list)
        write_to_file(data, save)


def write_to_file(data: List[str], output_file: str):
    with open(output_file, 'a') as f:
        for line in data:
            f.write("%s\n" % line)


def main():
    parser = argparse.ArgumentParser("Create a training file.")
    parser.add_argument("--data", help="Data file.", required=True)
    parser.add_argument("--save", help="Output file.", required=True)
    parser.add_argument("--context", help="Type of context to use (sent|para). Default: paragraph context.", default='para')
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
    create_data( args.data, args.save, args.context)


if __name__ == '__main__':
    main()
