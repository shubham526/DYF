from sklearn.model_selection import train_test_split
import argparse
from typing import List

def split_data(data, train_file, valid_file, test_file):
    with open(data) as f:
        lines = f.readlines()

    train, test = train_test_split(lines, test_size=0.3)
    val, test = train_test_split(test, test_size=0.5)

    print('Writing training data...')
    write_to_file(train, train_file)
    print('[Done].')

    print('Writing validation data...')
    write_to_file(val, valid_file)
    print('[Done].')

    print('Writing test data...')
    write_to_file(test, test_file)
    print('[Done].')


def write_to_file(data: List[str], output_file: str):
    with open(output_file, 'w') as f:
        for line in data:
            f.write("%s" % line)

def main():
    parser = argparse.ArgumentParser("Script to split data into train, validation and test sets.")
    parser.add_argument('--data', help='Data file.')
    parser.add_argument('--train', help='Train Data file.')
    parser.add_argument('--val', help='Validation Data file.')
    parser.add_argument('--test', help='Test Data file.')
    args = parser.parse_args()
    split_data(args.data, args.train, args.val, args.test)


if __name__ == '__main__':
    main()
