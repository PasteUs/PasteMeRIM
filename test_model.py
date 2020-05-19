import unittest
import json
import pandas as pd
import data_preprocess
from data_preprocess.data_preprocess import extract_chinese
from config import *
from tensorflow import keras

data_preprocess.extract_chinese = extract_chinese

with open('resources/word2idx/word2idx.json') as file:
    word2idx = json.load(file)


def df_preprocess(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    df = data_preprocess.tokenize(df)
    return data_preprocess.tokens_to_ids(df, word2idx)


class MyModelTestCase(unittest.TestCase):
    def test_something(self):
        df: pd.DataFrame = pd.read_csv('resources/dataset/permanent_chinese_only_with_label.csv')
        df['label'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)
        df = df_preprocess(df)
        model: keras.Sequential = keras.models.load_model('resources/saved_models/PasteMeRIM')
        input_data = keras.preprocessing.sequence.pad_sequences(df['text'].values, maxlen=MAX_LENGTH)
        df['y_hat'] = model.predict(input_data)
        df.to_csv('test_result.csv', index=False)
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
