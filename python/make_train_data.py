import json
import gzip
import sys
import tqdm
import os
import argparse
import utils
from joblib import Parallel, delayed
from typing import List, Dict, Set, Tuple
from pykson import Pykson, JsonObject, StringField, IntegerField, ListField, ObjectListField, ObjectField, Pykson, \
    BooleanField
from object_models import Location, Entity, AnnotatedText, AspectLinkExample, Aspect, Context

processor = utils.TextProcessor()
d = []
totals = {
    'nanni-test.jsonl.gz': 18289,
    'overly-frequent.jsonl.gz': 429160,
    'test.jsonl.gz': 4967,
    'train-remaining.jsonl.gz': 544892,
    'train-small.jsonl.gz': 5498,
    'validation.jsonl.gz': 4313

}


def to_pairwise_data(query: str, doc_pos: str, doc_neg_list: List[Tuple[str, str]]):
    data = []
    documents: Set[Tuple[str, str]]  = {
        (doc_pos, doc_neg) for _, doc_neg in doc_neg_list
    }

    for doc_pos, doc_neg in documents:
        data.append(
            json.dumps({
                'query': query,
                'doc_pos': doc_pos,
                'doc_neg': doc_neg
            })
        )
        return data


    # data.append(
    #     json.dumps({
    #         'query': query,
    #         'doc_pos': doc_pos,
    #         'doc_neg': doc_neg
    #     })
    #     for doc_pos, doc_neg in documents
    # )
    # return [
    #     json.dumps({
    #         'query': query,
    #         'doc_pos': doc_pos,
    #         'doc_neg': doc_neg
    #     })
    #     for doc_pos, doc_neg in documents
    # ]


def to_pointwise_data(query: str, doc_pos: str, doc_neg_list: List[Tuple[str, str]], data: List[str]) -> None:

    data.append(json.dumps({
        'query': query,
        'doc': doc_pos,
        'label': 1
    }))


    # data: List[str] = [json.dumps({
    #     'query': query,
    #     'doc': doc_pos,
    #     'label': 1
    # })]
    for _, doc_neg in doc_neg_list:
        data.append(json.dumps({
            'query': query,
            'doc': doc_neg,
            'label': 0
        }))
    #
    # return data


def to_pairwise_data(example: AspectLinkExample):
    data: List[str] = []
    query: str = processor.preprocess(example.context.paragraph.content)
    doc_pos: str = utils.get_positive_doc(example.candidate_aspects, example.true_aspect)
    doc_neg_list: List[Tuple[str, str]] = utils.get_negative_doc_list(example.candidate_aspects, example.true_aspect)
    documents: Set[Tuple[str, str]] = {
        (doc_pos, doc_neg) for _, doc_neg in doc_neg_list
    }

    for doc_pos, doc_neg in documents:
        data.append(
            json.dumps({
                'query': query,
                'doc_pos': doc_pos,
                'doc_neg': doc_neg
            })
        )
    return data


def to_pointwise_data(example: AspectLinkExample):
    data: List[str] = []
    query: str = processor.preprocess(example.context.paragraph.content)
    doc_pos: str = utils.get_positive_doc(example.candidate_aspects, example.true_aspect)
    doc_neg_list: List[Tuple[str, str]] = utils.get_negative_doc_list(example.candidate_aspects, example.true_aspect)
    data.append(json.dumps({
        'query': query,
        'doc': doc_pos,
        'label': 1
    }))
    for _, doc_neg in doc_neg_list:
        data.append(json.dumps({
            'query': query,
            'doc': doc_neg,
            'label': 0
        }))
    return data



def create_data(data_type: str, data_file: str, save: str, context_type: str, num_workers: int) -> None:
    data = Parallel(n_jobs=2, verbose=1, backend='multiprocessing')(delayed(do_stuff)(example) for example in utils.aspect_link_examples(data_file))
    print(len(data))
    print(type(data))
    # print('Creating {} data.'.format(data_type))
    # total = totals[os.path.basename(data_file)]
    # # print('Reading data...')
    # # examples: List[str] = utils.read_data(data_file, total)
    # # print('[Done].')
    # data: List[str] = []
    # for example in tqdm.tqdm(utils.aspect_link_examples(data_file), total=total):
    # #for example in tqdm.tqdm(examples, total=total):
    #     query: str = processor.preprocess(example.context.sentence.content) if context_type == 'sent' else processor.preprocess(example.context.paragraph.content)
    #     doc_pos: str = utils.get_positive_doc(example.candidate_aspects, example.true_aspect)
    #     doc_neg_list: List[Tuple[str, str]] = utils.get_negative_doc_list(example.candidate_aspects, example.true_aspect)
    #     to_pointwise_data(query, doc_pos, doc_neg_list, data) if data_type == 'pointwise' else to_pairwise_data(query, doc_pos, doc_neg_list, data)

    write_to_file(data, save)


def write_to_file(data: List[str], output_file: str):
    with open(output_file, 'a') as f:
        for line in data:
            f.write("%s\n" % line)


def main():
    parser = argparse.ArgumentParser("Create a training file.")
    parser.add_argument("--mode", help="Type of data (pairwise|pointwise).", required=True)
    parser.add_argument("--data", help="Data file.", required=True)
    parser.add_argument("--save", help="Output file.", required=True)
    parser.add_argument("--context", help="Type of context to use (sent|para). Default: paragraph context.", default='para')
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
    create_data(args.mode, args.data, args.save, args.context)


if __name__ == '__main__':
    main()
