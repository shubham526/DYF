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
from scipy import spatial
import operator
import torch

totals = {
    'nanni-test.jsonl.gz': 18289,
    'overly-frequent.jsonl.gz': 429160,
    'test.jsonl.gz': 4967,
    'train-remaining.jsonl.gz': 544892,
    'train-small.jsonl.gz': 5498,
    'validation.jsonl.gz': 4313

}


def cosine_similarity(emb1, emb2):
    return 1 - spatial.distance.cosine(emb1, emb2)

def write_to_file(context_id, aspect_scores, save):
    rank = 1
    with open(save, 'a') as f:
        for aspect_id, score in aspect_scores.items():
            if score != 0:
                run_file_string = context_id + ' Q0 ' + aspect_id + ' ' + str(rank) + ' ' +str(score) + ' EntityEmb'
                f.write("%s\n" % run_file_string)
                rank += 1

def aspect_score(context_entities, aspect_entities, embeddings) -> float:
    aspect_entity_scores: List[float] = []
    for aspect_entity in aspect_entities:
        if aspect_entity in embeddings:
            score: float = 0
            for context_entity in context_entities:
                if context_entity in embeddings:
                    score += cosine_similarity(embeddings[aspect_entity], embeddings[context_entity])
            aspect_entity_scores.append(score)

    return sum(aspect_entity_scores)


def score_aspect(data_file: str, save: str, context_type: str, embeddings: Dict[str, Any]):
    total = totals[os.path.basename(data_file)]
    for example in tqdm(utils.aspect_link_examples(data_file), total=total):
        context_id  = example.id
        score_dict = {}
        # if context_id in embeddings:
        #     example_embeddings = embeddings[context_id]
        #     context_entities = utils.get_entity_ids_only(
        #     example.context.sentence.entities) if context_type == 'sent' else utils.get_entity_ids_only(
        #     example.context.paragraph.entities)
        #     candidate_aspects = example.candidate_aspects
        #     for aspect in candidate_aspects:
        #         aspect_id = aspect.aspect_id
        #         aspect_entities = utils.get_entity_ids_only(aspect.aspect_content.entities)
        #         score = aspect_score(
        #             context_entities=set(context_entities),
        #             aspect_entities=set(aspect_entities),
        #             embeddings=example_embeddings
        #         )
        #         score_dict[aspect_id] = score

        context_entities = utils.get_entity_ids_only(
            example.context.sentence.entities) if context_type == 'sent' else utils.get_entity_ids_only(
            example.context.paragraph.entities)
        candidate_aspects = example.candidate_aspects
        for aspect in candidate_aspects:
            aspect_id = aspect.aspect_id
            aspect_entities = utils.get_entity_ids_only(aspect.aspect_content.entities)
            score = aspect_score(
                context_entities=set(context_entities),
                aspect_entities=set(aspect_entities),
                embeddings=embeddings
            )
            score_dict[aspect_id] = score
        sorted_score_dict = dict(sorted(score_dict.items(), key=operator.itemgetter(1), reverse=True))
        write_to_file(context_id, sorted_score_dict, save)

def main():
    parser = argparse.ArgumentParser("Rank aspects using entity embeddings.")
    parser.add_argument("--data", help="Data file.", required=True, type=str)
    parser.add_argument("--save", help="Output file.", required=True, type=str)
    parser.add_argument("--context", help="Type of context to use (sent|para). Default: paragraph context.",
                        default='para', type=str)
    parser.add_argument("--embeddings", help="Entity embedding file.", type=str)
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])

    device = torch.device(cuda_device if torch.cuda.is_available() else 'cpu')
    embeddings = torch.load(args.embeddings, map_location=device)

    # print('Loading entity embeddings...')
    # with open(args.embeddings, 'r') as f:
    #     embeddings = json.load(f)
    # print('[Done].')

    score_aspect(data_file=args.data, save=args.save, context_type=args.context, embeddings=embeddings)


if __name__ == '__main__':
    main()