import typing
import numpy as np
from tensorflow import keras


def get_embedding_weights_info_from_file(file_path: str) -> (int, int):
    with open(file_path) as file:
        first_line: str = file.readline()
        word_count, dimension = map(int, first_line.split())
        return word_count, dimension


def load_embedding_weights_and_word2idx(file_path: str) -> (np.ndarray, typing.Dict[str, int]):
    """
    get embedding weights and word to index from https://github.com/Embedding/Chinese-Word-Vectors
    :param file_path: embedding weights file path
    :return: embedding_weights: np.array, word2idx: dict[str, int]
    """
    word_count, dimension = get_embedding_weights_info_from_file(file_path=file_path)

    embedding_weights: np.ndarray = np.zeros([word_count + 1, dimension])
    word2idx: typing.Dict[str, int] = {}
    idx2word: typing.List[str] = ['OOV']

    with open(file_path) as file:
        for index, line in enumerate(file):
            if index > 0:
                split_line: typing.List[str] = line.split()
                word: str = split_line[0]
                vector: np.ndarray = np.array(split_line[1:], dtype=np.float32)
                embedding_weights[index] = vector
                word2idx[word] = index
                idx2word.append(word)

    return embedding_weights, word2idx


def get_embedding_layer(embedding_weights: np.ndarray, max_length: int) -> keras.layers.Layer:
    """
    generate keras embedding layer from np.array
    :param embedding_weights: embedding weights
    :param max_length: max tokens length
    :return: embedding layer
    """
    return keras.layers.Embedding(
        input_dim=embedding_weights.shape[0],
        output_dim=embedding_weights.shape[1],
        weights=[embedding_weights],
        input_length=max_length,
        trainable=False
    )
