import json
import gzip
import sys
import tqdm
import os
import argparse
import utils
from typing import List, Set, Any, Dict

seen: Set[str] = set()


def to_doc(candidate_aspects) -> List[str]:
    data: List[str] = []
    for aspect in candidate_aspects:
        doc_id: str = aspect['aspect_id']
        if doc_id not in seen:
            doc: str = aspect['aspect_content']['content']
            data.append(json.dumps({
                'doc_id': doc_id,
                'doc': doc
            }))
            data.append("\n")
            seen.add(doc_id)
    return data


def create_data(data_file: str, out_file: str) -> None:
    for example in tqdm.tqdm(utils.aspect_link_examples(data_file), total=7893275):
        candidate_aspects = example['candidate_aspects']
        data: List[str] = to_doc(candidate_aspects)
        write_to_file(data, out_file)


def write_to_file(data, output_file):
    file = open(output_file, "a")
    file.writelines(data)
    file.close()


def main():
    parser = argparse.ArgumentParser("Create document file.")
    parser.add_argument("--data", help="Data file.", required=True)
    parser.add_argument("--save", help="Output file.", required=True)
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
    create_data(args.data, args.save)


if __name__ == '__main__':
    main()
