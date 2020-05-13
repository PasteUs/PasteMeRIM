TEST_MODE = True  # Using test file or not
MAX_LENGTH = 128  # Sequence padding length
EPOCHS = 1
BATCH_SIZE = 32
MODEL_SAVING_PATH = 'resources/saved_models/model'


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

