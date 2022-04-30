import json
import gzip
import sys
from tqdm import tqdm
import os
import argparse
import utils
from typing import List, Dict, Set, Tuple, Any

processor = utils.TextProcessor()

def make_vocab(entities: List[str], id2name: Dict[str, str]):
    return [processor.preprocess(id2name[e]) for e in entities if e in id2name.keys()]
    # for e in entities:
    #     if e in id2name.keys():
    #         s.append(processor.preprocess(id2name[e]))
    # return s

def load_id2name(file: str) -> Dict[str, str]:
    res: Dict[str, str] = {}
    with open(file, 'r') as f:
        for line in f:
            parts = line.split('\t')
            key = parts[0]
            value = parts[1]
            res[key] = value
    return res


def load_entities(file: str) -> List[str]:
    res = []
    with open(file, 'r') as f:
        for line in f:
            line = line.strip('\n')
            res.append(line)
    return res

def write_to_file(data: List[str], save: str):
    with open(save, 'a') as f:
        for line in data:
            f.write("%s\n" % line)

def main():
    parser = argparse.ArgumentParser("Create entity vocab for EDRM.")
    parser.add_argument("--entities", help="Data file.", required=True, type=str)
    parser.add_argument("--save", help="Output file.", required=True, type=str)
    parser.add_argument("--id2name", help="EntityId-->EntityName mappings.", required=True, type=str)
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])

    print('Loading entities..')
    entities: List[str] = load_entities(args.entities)
    print('[Done].')

    print('Loading EntityId --> EntityName mappings...')
    id2name: Dict[str, str] = load_id2name(args.id2name)
    print('[Done].')

    print('Making vocab...')
    s = make_vocab(entities, id2name)
    print('[Done].')

    print('Writing to file...')
    write_to_file(s, args.save)
    print('[Done].')


if __name__ == '__main__':
    main()






