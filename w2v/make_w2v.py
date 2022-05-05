import numpy as np
from gensim.models import KeyedVectors


TOP_WORDS = 50000
W2V_FILEPATH = "/home/s2110184/Model/nwjc_word_skip_200_8_25_0_1e4_6_1_0_15.txt.vec"


wv: KeyedVectors
wv = KeyedVectors.load_word2vec_format(W2V_FILEPATH, binary=False)
model = KeyedVectors(200)


def top_key_weights(top_count=30000):
    top_count = top_count - 4
    top_keys = [wv.index_to_key[i] for i in range(top_count)]
    top_weights = [wv[key] for key in top_keys]
    return top_keys, top_weights


def add_special_chars(top_keys, top_weights):
    special_char = ["<pad>", "<s>", "</s>", "<unk>"]
    special_weights = [
        np.zeros((200), dtype=np.float32),
        np.ones((200), dtype=np.float32),
        np.random.randn((200)),
        np.random.randn((200)),
    ]

    model.add_vectors(
        keys=special_char + top_keys, weights=special_weights + top_weights
    )


def display_w2v(filepath, binary=True):
    new_model = KeyedVectors.load_word2vec_format(filepath, binary=binary)

    print("Loaded filepath : {}".format(filepath))
    for i in range(10):
        key = new_model.index_to_key[i]
        weight = new_model[key]

        print(f"{key} : ")
        print(weight)
        print("------------")


def main():
    top_keys, top_weights = top_key_weights(TOP_WORDS)
    add_special_chars(top_keys, top_weights)

    filepath = "top_{}".format(TOP_WORDS)
    model.save(filepath)
    model.save_word2vec_format(filepath + ".model", binary=True)

    display_w2v(filepath + ".model")


if __name__ == "__main__":
    main()
