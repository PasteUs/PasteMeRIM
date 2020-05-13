import os
import data_preprocess
import pandas as pd
import numpy as np
from training import embedding
from tensorflow import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Disables the warning, doesn't enable AVX/FMA

TEST_MODE = True  # Using test file or not
MAX_LENGTH = 4096  # Sequence padding length
EPOCHS = 32
BATCH_SIZE = 32


def get_file_path(test_mode: bool) -> (str, str):
    if test_mode:
        word2_vec_test_path = 'resources/word2vec/sgns.test.word'
        dataset_test_path = 'resources/dataset/permanent_chinese_only_with_label.test.csv'
        return word2_vec_test_path, dataset_test_path
    else:
        word2_vec_prod_path = 'resources/word2vec/sgns.weibo.word'
        dataset_prod_path = 'resources/dataset/permanent_chinese_only_with_label.csv'
        return word2_vec_prod_path, dataset_prod_path


WORD2VEC_PATH, DATASET_PATH = get_file_path(TEST_MODE)


def main():
    embedding_weights, word2idx = embedding.parse_word2vec(file_path=WORD2VEC_PATH)
    embedding_layer = embedding.get_embedding_layer(embedding_weights, MAX_LENGTH)

    df = pd.read_csv(DATASET_PATH)
    df['label'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)
    df = data_preprocess.tokenize(df)
    df = data_preprocess.tokens_to_ids(df, word2idx)
    df['text'] = keras.preprocessing.sequence.pad_sequences(df['text'].values, maxlen=MAX_LENGTH)

    split_point = len(df) * 70 // 100

    train_df = df[:split_point]
    valid_df = df[split_point:]

    train_df = data_preprocess.balanced_sampling(train_df)

    model = keras.Sequential([
        keras.layers.Input(shape=(None,), dtype=np.int32),
        embedding_layer,
        keras.layers.Bidirectional(keras.layers.LSTM(64)),
        keras.layers.Dense(64, activation=keras.activations.relu),
        keras.layers.Dense(1)
    ])

    model.compile(
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer=keras.optimizers.Adam(1e-4),
        metrics=['accuracy']
    )

    model.fit(
        x=train_df['text'],
        y=train_df['label'],
        validation_data=(valid_df['text'], valid_df['label']),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )


if __name__ == '__main__':
    main()
