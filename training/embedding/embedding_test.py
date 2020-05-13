import unittest
import numpy as np
from training.embedding import embedding


class MyEmbeddingTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(MyEmbeddingTestCase, self).__init__(*args, **kwargs)
        self.asset_path = '../../resources/word2vec/sgns.test.word'

    def test_get_embedding_info(self):
        word_count, dimension = embedding.get_embedding_weights_info_from_file(self.asset_path)
        self.assertEqual(word_count, 6)
        self.assertEqual(dimension, 2)

    def test_get_embedding_weights(self):
        word_count, dimension = embedding.get_embedding_weights_info_from_file(self.asset_path)
        embedding_weights, word2idx = embedding.load_embedding_weights_and_word2idx(self.asset_path)
        self.assertEqual(embedding_weights.shape, (word_count + 1, dimension))
        weights: np.ndarray = np.array([[0.000000, 0.000000],
                                        [0.102387, 0.111146],
                                        [0.081348, 0.073545],
                                        [0.155037, 0.084531],
                                        [0.096599, 0.089405],
                                        [0.146402, 0.137529],
                                        [0.176425, 0.055730]])
        equal_result: np.ndarray = np.abs(weights - embedding_weights) < 1e-7
        self.assertTrue(equal_result.all())


if __name__ == '__main__':
    unittest.main()
