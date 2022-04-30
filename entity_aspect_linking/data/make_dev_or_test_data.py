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


def to_data(example: AspectLinkExample, context_type: str, features: Dict[str, Dict[str, str]]) -> List[str]:
    data: List[str] = []

    query_id: str = example.id
    feature_dict: Dict[str, str] = features[query_id]
    if query_id in features:
        query: Dict[str, Any] = {
            'text': example.context.sentence.content if context_type == 'sent' else example.context.paragraph.content,
            'entities': utils.get_entity_ids_only(
                example.context.sentence.entities) if context_type == 'sent' else utils.get_entity_ids_only(
                example.context.paragraph.entities)
        }
        # query: str = processor.preprocess(example.context.sentence.content) if context_type == 'sent' else processor.preprocess(example.context.paragraph.content)

        doc_pos: str = utils.get_positive_doc(example.candidate_aspects, example.true_aspect, feature_dict)
        doc_pos_id: str = example.true_aspect
        doc_neg_list: List[Tuple[str, Dict[str, Any]]] = utils.get_negative_doc_list(example.candidate_aspects, example.true_aspect, feature_dict)

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

# def to_data(
#         example: AspectLinkExample,
#         context_type: str,
#         entities: Dict[str, Dict[str, str]],
#         wiki2vec: Dict[str, List[float]]
# ) -> List[str]:
#     data: List[str] = []
#     query_id: str = example.id
#     if query_id in entities:
#         entity_data: Dict[str, str] = entities[query_id]
#         query = utils.get_query(
#             context=example.context,
#             context_type=context_type,
#             entity_data=entity_data,
#             wiki2vec=wiki2vec
#         )
#         doc_pos: Dict[str, Any] = utils.get_positive_doc(
#             candidate_aspects=example.candidate_aspects,
#             true_aspect=example.true_aspect,
#             entity_data=entity_data,
#             wiki2vec=wiki2vec
#         )
#         doc_neg_list: List[Tuple[str, Dict[str, Any]]] = utils.get_negative_doc_list(
#             candidate_aspects=example.candidate_aspects,
#             true_aspect=example.true_aspect,
#             entity_data=entity_data,
#             wiki2vec=wiki2vec
#         )
#         if doc_pos and len(doc_neg_list) >= 1 and len(query['entities']) != 0:
#             data.append(json.dumps({
#                 'query': query,
#                 'doc': doc_pos,
#                 'label': 1,
#                 'query_id': query_id,
#                 'doc_id': example.true_aspect
#             }))
#
#             for doc_neg_id, doc_neg in doc_neg_list:
#                 data.append(json.dumps({
#                     'query': query,
#                     'doc': doc_neg,
#                     'label': 0,
#                     'query_id': query_id,
#                     'doc_id': doc_neg_id
#                 }))
#         return data
#     else:
#         return []


def create_data(data_file: str, save: str, context_type: str, features: Dict[str, Dict[str, str]], num_workers: int) -> None:
    print('Context type: {}'.format(context_type))
    print('Number of processes = {}'.format(num_workers))
    total = totals[os.path.basename(data_file)]
    with tqdm_joblib(tqdm(desc="Progress", total=total)) as progress_bar:
        data = Parallel(n_jobs=num_workers, backend='multiprocessing')(
            delayed(to_data)(example, context_type, features) for example in utils.aspect_link_examples(data_file))

    print('Writing to file...')
    for d in data:
        write_to_file(d, save)
    print('[Done].')
    print('File written to ==> {}'.format(save))

# def create_data(
#         data_file: str,
#         save: str,
#         context_type: str,
#         num_workers: int,
#         entities: Dict[str, Dict[str, str]],
#         wiki2vec: Dict[str, List[float]]
# ) -> None:
#
#
#     print('Context type: {}'.format(context_type))
#     print('Number of processes = {}'.format(num_workers))
#     total = totals[os.path.basename(data_file)]
#
#     with tqdm_joblib(tqdm(desc="Progress", total=total)) as progress_bar:
#         data = Parallel(n_jobs=num_workers, backend='multiprocessing')(
#             delayed(to_data)(example, context_type, entities, wiki2vec)
#             for example in utils.aspect_link_examples(data_file)
#         )
#
#     print('Writing to file...')
#     for d in data:
#         write_to_file(d, save)
#     print('[Done].')
#     print('File written to ==> {}'.format(save))


def write_to_file(data: List[str], output_file: str):
    with open(output_file, 'a') as f:
        for line in data:
            f.write("%s\n" % line)


def read_feature_file(feature_file: str) -> Dict[str, Dict[str, str]]:
    feature_dict = {}
    with open(feature_file, 'r') as f:
        for line in f:
            line_parts = line.split('\t')
            query_id = line_parts[0].strip()
            doc_id = line_parts[1].strip()
            feature = line_parts[2].strip()
            fet = feature_dict[query_id] if query_id in feature_dict else {}
            fet[doc_id] = feature
            feature_dict[query_id] = fet
    return feature_dict


def main():
    parser = argparse.ArgumentParser("Create a validation or test file.")
    parser.add_argument("--data", help="Data file.", required=True)
    parser.add_argument("--save", help="Output file.", required=True)
    parser.add_argument("--context", help="Type of context to use (sent|para). Default: paragraph context.",
                        default='para')
    parser.add_argument("--features", help="Feature file.", required=True, type=str)
    parser.add_argument("--num-workers", help="Number of processes to use. Default: 4.",
                        default=4, type=int)
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
    print('Reading features...')
    features: Dict[str, Dict[str, str]] = read_feature_file(args.features)
    print('[Done].')

    create_data(args.data, args.save, args.context, features, args.num_workers)


# def main():
#     parser = argparse.ArgumentParser("Create a validation or test file.")
#     parser.add_argument("--data", help="Data file.", required=True)
#     parser.add_argument("--save", help="Output file.", required=True)
#     parser.add_argument("--context", help="Type of context to use (sent|para). Default: paragraph context.",
#                         default='para')
#     parser.add_argument("--num-workers", help="Number of processes to use. Default: 4.",
#                         default=4, type=int)
#     parser.add_argument("--entity-data", help="File containing entity data (BM25Psg or LeadText of entities).",
#                         required=True, type=str)
#     parser.add_argument("--wiki2vec", help="File containing Wiki2Vec embeddings for entities.",
#                         required=True, type=str)
#     args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
#
#     print('Loading entity data...')
#     entities: Dict[str, Dict[str, str]] = utils.load_entity_data(args.entity_data)
#     print('[Done].')
#
#     print('Loading Wiki2Vec embeddings...')
#     wiki2vec: Dict[str, List[float]] = utils.load_wiki2vec(args.wiki2vec)
#     print('[Done].')
#
#     create_data( args.data, args.save, args.context, args.num_workers, entities, wiki2vec)


if __name__ == '__main__':
    main()
