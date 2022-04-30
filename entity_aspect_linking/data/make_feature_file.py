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

def read_feature_file(feature_file: str) -> Dict[str, Dict[str, List[float]]]:

    features: Dict[str, Dict[str, List[float]]] = {}

    with open(feature_file, 'r') as f:
        for line in f:
            line_parts = line.split()
            query_id: str = line_parts[-2][1:].strip()
            doc_id: str = line_parts[-1].strip()
            n: int = len(line_parts) - 2
            feature: List[float] = [float(line_parts[i].split(":")[1]) for i in range(2, n)]
            feature_dict: Dict[str, List[float]] = features[query_id] if query_id in features else {}
            feature_dict[doc_id] = feature
            features[query_id] = feature_dict

    return features


def write_to_file(features: Dict[str, Dict[str, List[float]]], save: str):
    with open(save, 'w') as f:
        for query_id, feature_dict in features.items():
            for doc_id, feature in feature_dict.items():
                f.write("%s\t%s\t%s\n" % (query_id, doc_id, json.dumps(feature)))



def main():
    parser = argparse.ArgumentParser("Create a feature file.")
    parser.add_argument("--features", help="Feature file.", required=True, type=str)
    parser.add_argument("--save", help="Output file.", required=True, type=str)
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])

    print('Reading feature file...')
    features = read_feature_file(args.features)
    print('[Done].')

    print('Saving to file...')
    write_to_file(features, args.save)
    print('[Done].')


if __name__ == '__main__':
    main()


