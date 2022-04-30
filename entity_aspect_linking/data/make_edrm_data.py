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

def get_entities_or_desc(entities: List[str], data: Dict[str, str]) -> List[str]:
    return [processor.preprocess(data[entity_id]) for entity_id in entities if entity_id in data]


def get_query_data(
        context: AnnotatedText,
        id2name: Dict[str, str],
        desc: Dict[str, str]
) -> Tuple[str, List[str], List[str]]:

    query: str = processor.preprocess(context.content)
    query_entity_ids: List[str] = list(set(utils.get_entity_ids_only(context.entities)))
    query_ent: List[str] = get_entities_or_desc(
        entities=query_entity_ids,
        data=id2name
    )
    query_des: List[str] = get_entities_or_desc(
        entities=query_entity_ids,
        data=desc
    )

    return query, query_ent, query_des


def get_doc_pos_data(
        candidate_aspects: List[Aspect],
        true_aspect: str,
        id2name: Dict[str, str],
        desc: Dict[str, str]
) -> Tuple[str, List[str], List[str]]:

    doc_pos_dict: Dict[str, Any] = utils.get_positive_doc(candidate_aspects, true_aspect)

    if doc_pos_dict:
        doc_pos: str = processor.preprocess(doc_pos_dict['text'])
        doc_pos_ent: List[str] = get_entities_or_desc(
            entities=list(set(doc_pos_dict['entities'])),
            data=id2name
        )
        doc_pos_desc: List[str] = get_entities_or_desc(
            entities=list(set(doc_pos_dict['entities'])),
            data=desc
        )
        return doc_pos, doc_pos_ent, doc_pos_desc
    return ' ', [], []


def get_doc_neg_data(
        candidate_aspects: List[Aspect],
        true_aspect: str,
        id2name: Dict[str, str],
        desc: Dict[str, str]
) -> Tuple[List[str], List[str], List[List[str]], List[List[str]]]:

    doc_neg_data_list: List[Tuple[str, Dict[str, Any]]] = utils.get_negative_doc_list(
        candidate_aspects=candidate_aspects,
        true_aspect=true_aspect
    )

    if len(doc_neg_data_list) != 0:

        doc_neg_list: List[str] = [doc['text'] for _, doc in doc_neg_data_list]
        doc_neg_id_list: List[str] = [doc_id for doc_id, _ in doc_neg_data_list]
        doc_neg_ent_list: List[List[str]] = [
            get_entities_or_desc(
                entities=list(set(doc['entities'])),
                data=id2name
            )
            for _, doc in doc_neg_data_list
        ]
        doc_neg_desc_list: List[List[str]] = [
            get_entities_or_desc(
                entities=list(set(doc['entities'])),
                data=desc
            )
            for _, doc in doc_neg_data_list
        ]
        return doc_neg_list, doc_neg_id_list, doc_neg_ent_list, doc_neg_desc_list
    return [], [], [], []


def get_data(
        context: AnnotatedText,
        candidate_aspects: List[Aspect],
        true_aspect: str,
        id2name: Dict[str, str],
        desc: Dict[str, str]
) -> Dict[str, Any]:

    query, query_ent, query_des = get_query_data(
        context=context,
        id2name=id2name,
        desc=desc
    )

    doc_pos, doc_pos_ent, doc_pos_des = get_doc_pos_data(
        candidate_aspects=candidate_aspects,
        true_aspect=true_aspect,
        id2name=id2name,
        desc=desc
    )

    doc_neg_list, doc_neg_id_list, doc_neg_ent_list, doc_neg_desc_list = get_doc_neg_data(
        candidate_aspects=candidate_aspects,
        true_aspect=true_aspect,
        id2name=id2name,
        desc=desc
    )

    if doc_pos and len(doc_pos_ent) != 0 and len(doc_pos_des) != 0 and len(doc_neg_list) != 0 and len(doc_neg_id_list) != 0 and len(doc_neg_ent_list) != 0 and len(doc_neg_desc_list) != 0:
        return {
            'query': query,
            'query_ent': query_ent,
            'query_des': query_des,
            'doc_pos': doc_pos,
            'doc_pos_ent': doc_pos_ent,
            'doc_pos_des': doc_pos_des,
            'doc_neg_list': doc_neg_list,
            'doc_neg_id_list': doc_neg_id_list,
            'doc_neg_ent_list': doc_neg_ent_list,
            'doc_neg_des_list': doc_neg_desc_list
        }
    return {}


def to_train_data(
        mode: str,
        example: AspectLinkExample,
        context_type: str,
        id2name: Dict[str, str],
        desc: Dict[str, Dict[str, str]],
        data: List[str]
):

    # data: List[str] = []
    query_id: str = example.id
    context: AnnotatedText = example.context.sentence if context_type == 'sent' else example.context.paragraph

    if query_id in desc:

        d: Dict[str, Any] = get_data(
            context=context,
            candidate_aspects=example.candidate_aspects,
            true_aspect=example.true_aspect,
            id2name=id2name,
            desc=desc[query_id]
        )

        if d:
            if mode == 'pairwise':
                for doc_neg, doc_neg_ent, doc_neg_desc in zip(d['doc_neg_list'], d['doc_neg_ent_list'], d['doc_neg_des_list']):
                    data.append(
                        json.dumps({
                            'query': d['query'],
                            'query_ent': d['query_ent'],
                            'query_des': d['query_des'],
                            'doc_pos': d['doc_pos'],
                            'doc_pos_ent': d['doc_pos_ent'],
                            'doc_pos_des': d['doc_pos_des'],
                            'doc_neg': doc_neg,
                            'doc_neg_ent': doc_neg_ent,
                            'doc_neg_des': doc_neg_desc
                        })
                    )
            else:
                data.append(
                    json.dumps({
                        'query': d['query'],
                        'query_ent': d['query_ent'],
                        'query_des': d['query_des'],
                        'doc': d['doc_pos'],
                        'doc_ent': d['doc_pos_ent'],
                        'doc_des': d['doc_pos_des'],
                        'label': 1
                    })
                )
                for doc_neg, doc_neg_ent, doc_neg_desc in zip(d['doc_neg_list'], d['doc_neg_ent_list'], d['doc_neg_des_list']):
                    data.append(
                        json.dumps({
                            'query': d['query'],
                            'query_ent': d['query_ent'],
                            'query_des': d['query_des'],
                            'doc': doc_neg,
                            'doc_ent': doc_neg_ent,
                            'doc_des': doc_neg_desc,
                            'label': 0
                        })
                    )

        # return data


def to_dev_or_test_data(
        example: AspectLinkExample,
        context_type: str,
        id2name: Dict[str, str],
        desc: Dict[str, Dict[str, str]],
        data: List[str]
):

    #data: List[str] = []
    query_id: str = example.id
    context: AnnotatedText = example.context.sentence if context_type == 'sent' else example.context.paragraph

    if query_id in desc:

        d: Dict[str, Any] = get_data(
            context=context,
            candidate_aspects=example.candidate_aspects,
            true_aspect=example.true_aspect,
            id2name=id2name,
            desc=desc[query_id]
        )

        if d:
            data.append(
                json.dumps({
                    'query': d['query'],
                    'query_id': query_id,
                    'query_ent': d['query_ent'],
                    'query_des': d['query_des'],
                    'doc': d['doc_pos'],
                    'doc_id': example.true_aspect,
                    'doc_ent': d['doc_pos_ent'],
                    'doc_des': d['doc_pos_des'],
                    'label': 1,
                    'retrieval_score': 0.0
                })
            )
            for doc_neg, doc_neg_id, doc_neg_ent, doc_neg_desc in zip(d['doc_neg_list'], d['doc_neg_id_list'], d['doc_neg_ent_list'], d['doc_neg_des_list']):
                data.append(
                    json.dumps({
                        'query': d['query'],
                        'query_ent': d['query_ent'],
                        'query_des': d['query_des'],
                        'doc': doc_neg,
                        'doc_id': doc_neg_id,
                        'doc_ent': doc_neg_ent,
                        'doc_des': doc_neg_desc,
                        'label': 0,
                        'retrieval_score': 0.0
                    })
                )


    # return data



def create_data(
        data_type: str,
        data_file: str,
        save: str,
        context_type: str,
        id2name: Dict[str, str],
        desc: Dict[str, Dict[str, str]],
) -> None:

    print('Data type: {}'.format(data_type))
    print('Context type: {}'.format(context_type))
    # print('Number of processes = {}'.format(num_workers))
    total = totals[os.path.basename(data_file)]
    data: List[str] = []

    if data_type == 'pairwise' or data_type == 'pointwise':
        for example in tqdm(utils.aspect_link_examples(data_file),total=total):
            to_train_data(data_type, example, context_type, id2name, desc, data)

    elif data_type == 'dev' or data_type == 'test':
        for example in tqdm(utils.aspect_link_examples(data_file), total=total):
            to_dev_or_test_data(example, context_type, id2name, desc, data)


        # with tqdm_joblib(tqdm(desc="Progress", total=total)) as progress_bar:
        #     data = Parallel(n_jobs=num_workers, backend='multiprocessing')(
        #         delayed(to_train_data)(data_type, example, context_type, id2name, desc) for example in utils.aspect_link_examples(data_file))

    # elif data_type == 'dev' or data_type == 'test':
    #     with tqdm_joblib(tqdm(desc="Progress", total=total)) as progress_bar:
    #         data = Parallel(n_jobs=num_workers, backend='multiprocessing')(
    #             delayed(to_dev_or_test_data)(example, context_type, id2name, desc) for example in
    #             utils.aspect_link_examples(data_file))


    print('Writing to file...')
    write_to_file(data, save)
    # for d in data:
    #     write_to_file(d, save)
    print('[Done].')
    print('File written to ==> {}'.format(save))


def write_to_file(data: List[str], output_file: str):
    with open(output_file, 'a') as f:
        for line in data:
            f.write("%s\n" % line)


def load_id2name(file: str) -> Dict[str, str]:
    res: Dict[str, str] = {}
    with open(file, 'r') as f:
        for line in f:
            parts = line.split('\t')
            key = parts[0]
            value = parts[1]
            res[key] = value
    return res

def load_desc(file_path: str) -> Dict[str, Dict[str, str]]:
    res: Dict[str, Dict[str, str]] = {}
    with open(file_path, 'r') as file:
        for line in file:
            line_parts = line.split("\t")
            if len(line_parts) == 3:
                query_id: str = line_parts[0]
                entity_id: str = line_parts[1]
                if query_id not in res:
                    res[query_id] = {}
                doc: str = line_parts[2]
                res[query_id][entity_id] = doc

    return res


def main():
    parser = argparse.ArgumentParser("Create a training file.")
    parser.add_argument("--mode", help="Type of data (pairwise|pointwise|dev|test).", required=True, type=str)
    parser.add_argument("--data", help="Data file.", required=True, type=str)
    parser.add_argument("--save", help="Output file.", required=True, type=str)
    parser.add_argument("--context", help="Type of context to use (sent|para). Default: paragraph context.",
                        default='para', type=str)
    parser.add_argument("--id2name", help="EntityId-->EntityName mappings.", required=True, type=str)
    parser.add_argument("--desc", help="File containing entity descriptions.", required=True, type=str)
    # parser.add_argument("--num-workers", help="Number of processes to use. Default: 4.",
    #                     default=4, type=int)
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])

    print('Loading EntityId --> EntityName mappings...')
    id2name: Dict[str, str] = load_id2name(args.id2name)
    print('[Done].')

    print('Loading entity descriptions...')
    desc: Dict[str, Dict[str, str]] = load_desc(args.desc)
    print('[Done].')

    create_data(args.mode, args.data, args.save, args.context, id2name, desc)


if __name__ == '__main__':
    main()