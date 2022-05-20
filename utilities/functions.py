import glob
import json

from utilities.constant import TWITTER_CORPUS_FOLDER


def load_json(filepath):
    with open(filepath) as f:
        json_ = json.load(f)
    return json_


def train_val_test(all_size=0.7, train_size=0.7, val_size=0.7):
    all_files = list(glob.glob(TWITTER_CORPUS_FOLDER + "*.json"))
    all_files = all_files[0 : int(len(all_files) * all_size)]

    train_files = all_files[0 : int(len(all_files) * train_size)]
    other_files = all_files[int(len(all_files) * train_size) :]

    val_files = other_files[0 : int(len(other_files) * val_size)]
    test_files = other_files[int(len(other_files) * val_size) :]

    return train_files, val_files, test_files
