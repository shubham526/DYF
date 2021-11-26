import json
import gzip
import sys
import tqdm
import os
import argparse
from typing import List
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

def aspect_link_examples(json_file: str) -> AspectLinkExample:
    """
    Reads the JSON-L file in gzip format.
    Generates an AspectLinkExample in a lazy way (using yield).
    :param json_file: JSON-L file in gzip format.
    """
    with gzip.open(json_file, 'rt', encoding='UTF-8') as zipfile:
        for line in zipfile:
            example = Pykson().from_json(line, AspectLinkExample)
            yield example


def create_data(data_dir: str, data_file: str, out_file: str, context_type: str) -> None:
    total = totals[data_file]
    data: List[str] = []
    input_file = os.path.join(data_dir, data_file)
    print('Context Type:' + context_type)
    for example in tqdm.tqdm(aspect_link_examples(input_file), total=total):
        query_id: str = example.id
        query: str = example.context.sentence.content if context_type == 'sent' else example.context.paragraph.content
        data.append(json.dumps({
            'query_id' : query_id,
            'query' : query
        }))
        data.append("\n")
    write_to_file(data, out_file)

def write_to_file(data, output_file):
    file = open(output_file, "a")
    file.writelines(data)
    file.close()


def main():
    parser = argparse.ArgumentParser("Create queries file in OpenMatch format.")
    parser.add_argument("--data-dir", help="Directory where data is stored.", required=True)
    parser.add_argument("--data-file", help="(Zipped) data file.", required=True)
    parser.add_argument("--out", help="Output file.", required=True)
    parser.add_argument("--context-type", help="Type of context to use (sent|para).", required=True)
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
    create_data(args.data_dir, args.data_file, args.out, args.context_type)


if __name__ == '__main__':
    main()