import argparse
import glob
import os
import random

import numpy as np

from utils import get_module_logger


def split(data_dir):
    """
    Create three splits from the processed records. The files should be moved to new folders in the 
    same directory. This folder should be named train, val and test.

    args:
        - data_dir [str]: data directory, /home/workspace/data/waymo
    """
    
    # Split the data present in `/home/workspace/data/waymo/training_and_validation` into train and val sets.
    # You should move the files rather than copy because of space limitations in the workspace.
    
    # 80% for training. 10% for validation. 10% for testing
    paths = glob.glob(f'{data_dir}/*.tfrecord')
    paths.sort()
    random.seed(100)
    random.shuffle(paths)

    split_size_1 = int(0.80 * len(paths))
    split_size_2 = int(0.90 * len(paths))
    
    training_data  = paths[:split_size_1]
    validation_data  = paths[split_size_1:split_size_2]
    testing_data = paths[split_size_2:]

    training_dir = f'{data_dir}/../train'
    validation_dir = f'{data_dir}/../val'
    testing_dir = f'{data_dir}/../test'

    if not os.path.exists(training_dir):
        os.makedirs(training_dir)

    if not os.path.exists(validation_dir):
        os.makedirs(validation_dir)

    if not os.path.exists(testing_dir):
        os.makedirs(testing_dir)

    for path in training_data:
        file_name = os.path.basename(path)
        new_path = f'{training_dir}/{file_name}'
        os.rename(path, new_path)

    for path in validation_data:
        file_name = os.path.basename(path)
        new_path = f'{validation_dir}/{file_name}'
        os.rename(path, new_path)

    for path in testing_data:
        file_name = os.path.basename(path)
        new_path = f'{testing_dir}/{file_name}'
        os.rename(path, new_path)

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--data_dir', required=True,
                        help='data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.data_dir)