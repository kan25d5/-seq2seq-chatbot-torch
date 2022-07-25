import os
import optuna
import torch
import pytorch_lightning as pl
from typing import List
from dataloader.tanaka_dataloader import TanakaDataLoader
from models.seq2seq_transformer import Seq2Seq
from utilities.vocab import TanakaVocabs
from sklearn.model_selection import train_test_split
from dataloader.twitter_dataset import TwitterDataset
from dataloader.twitter_transform import TwitterTransform
from utilities.constant import MAXLEN, TOP_WORDS, TRAIN_SIZE, VAL_SIZE, CHAR2ID


def get_datasets(sentiment_type: str, pkl_list: List[str]) -> List[TwitterDataset]:
    check_pkl = all(os.path.exists(filepath) for filepath in pkl_list)

    if check_pkl:
        return _load_datasets(pkl_list)
    else:
        return _make_datasets(sentiment_type)


def _load_datasets(pkl_list):
    print("The pkl data of the dataset exists.")
    print("\t" + "execute _load_datasets()")

    all_datasets = [TwitterDataset(MAXLEN, TwitterTransform()) for _ in range(3)]
    all_datasets[0].load_corpus_pkl(pkl_list[0])
    all_datasets[1].load_corpus_pkl(pkl_list[1])
    all_datasets[2].load_corpus_pkl(pkl_list[2])

    return all_datasets


def _make_datasets(sentiment_type):
    from utilities.functions import load_json

    print("The pkl data of the dataset does not exist.")
    print("\t" + "execute _make_datasets()")

    dataset_X = []
    dataset_y = []

    filepath = "output/{}.json".format(sentiment_type)
    dialogue_json = load_json(filepath)

    for dialogue in dialogue_json:
        msg = dialogue[0]
        res = dialogue[1]

        if len(msg.split()) > MAXLEN:
            continue
        if len(res.split()) > MAXLEN:
            continue

        dataset_X.append(msg)
        dataset_y.append(res)

        X_train, X_other, y_train, y_other = train_test_split(
            dataset_X, dataset_y, test_size=(1 - TRAIN_SIZE)
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_other, y_other, test_size=(1 - VAL_SIZE)
        )

        train_dataset = TwitterDataset(MAXLEN, TwitterTransform())
        train_dataset.messages = X_train
        train_dataset.responses = y_train

        val_dataset = TwitterDataset(MAXLEN, TwitterTransform())
        val_dataset.messages = X_val
        val_dataset.responses = y_val

        test_dataset = TwitterDataset(MAXLEN, TwitterTransform())
        test_dataset.messages = X_test
        test_dataset.responses = y_test

        return train_dataset, val_dataset, test_dataset


def get_vocab(all_datasets: List[TwitterDataset], pkl_list: List[str]):
    check_pkl = all(os.path.exists(filepath) for filepath in pkl_list)
    if check_pkl:
        return _get_vocab_exists()
    else:
        return _get_vocab_no_exists(all_datasets, pkl_list)


def _get_vocab_exists():
    vocabs = TanakaVocabs(TOP_WORDS)
    if os.path.exists(CHAR2ID):
        vocabs.load_char2id_pkl(CHAR2ID)
        return vocabs
    else:
        raise ValueError("loaded dataset pkl, but char2id does not exist.")


def _get_vocab_no_exists(all_datasets, pkl_list):
    vocabs = TanakaVocabs(TOP_WORDS)
    if not os.path.exists(CHAR2ID):
        vocabs.fit(all_datasets)

    vocabs.transform(all_datasets)
    all_datasets[0].save_corpus_pkl(pkl_list[0])
    all_datasets[1].save_corpus_pkl(pkl_list[1])
    all_datasets[2].save_corpus_pkl(pkl_list[2])

    return vocabs


def get_dataloader(all_datasets: List[TwitterDataset], batch_size: int):
    from dataloader.tanaka_dataloader import TanakaDataLoader

    train_dataloader = TanakaDataLoader(all_datasets[0], batch_size=batch_size)
    val_dataloader = TanakaDataLoader(all_datasets[1], batch_size=batch_size)
    test_dataloader = TanakaDataLoader(all_datasets[2], batch_size=1, random_state=0)

    all_dataloader = [train_dataloader, val_dataloader, test_dataloader]
    return all_dataloader