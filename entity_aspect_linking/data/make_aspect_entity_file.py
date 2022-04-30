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

def write_to_file(data: Dict[str, List[str]], output_file: str) -> None:
    with open(output_file, 'w') as f:
        for key, value in data.items():
            f.write("%s\t%s\n" % (key, json.dumps(value)))

def read_tsv(file: str):
    with open(file, 'r') as f:
        return [line.split('\t')[0]  for line in f]


def get_all_aspect_entities(candidate_aspects):
    l = []
    for aspect in candidate_aspects:
        l.extend(utils.get_entity_ids_only(aspect.aspect_content.entities))
    return set(l)


def create_data(data_file: str, save: str, entity_list):
    total = totals[os.path.basename(data_file)]

    data: Dict[str, List[str]] = {}
    count = 0

    print('Creating data...')

    for example in tqdm(utils.aspect_link_examples(data_file), total=total):
        aspect_entities = get_all_aspect_entities(example.candidate_aspects)
        aspect_found = set(aspect_entities).intersection(set(entity_list))

        if len(aspect_found) > 0:
            data[example.id] = list(aspect_found)
        else:
            count += 1

    print('[Done].')

    if count > 0:
        print('No context entities for {} examples.'.format(count))

    print('Writing to file...')
    write_to_file(data, save)
    print('[Done].')


def main():
    parser = argparse.ArgumentParser("Create a mapping from ContextId --> List[AspectEntity].")
    parser.add_argument("--data", help="Data file.", required=True, type=str)
    parser.add_argument("--save", help="Output file.", required=True, type=str)
    parser.add_argument("--entity2psg", help="EntityId to ParaId mappings", required=True, type=str)
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])

    print('Getting entities from entity2psg file...')
    entity_list = read_tsv(args.entity2psg)
    create_data(data_file=args.data, save=args.save, entity_list=entity_list)


if __name__ == '__main__':
    main()




