"""
doccano 最大只能上传 1MB 的文件，需要将 CSV 拆分
"""

import re
import os
import pandas as pd
from zhon import hanzi

SUB_DIR_NAME = 'batch_filtered_permanent'
FILE_PREFIX = 'filtered_permanent'
BUF_FILE_NAME = '.buf.file'
SOURCE_FILE_NAME = 'filtered_permanent.csv'


def df_to_txt(raw_df: pd.DataFrame, file_path: str) -> None:
    df = raw_df.copy()
    zh_pattern = re.compile('[{}]'.format(hanzi.characters))
    df['text'] = df['text'].apply(lambda x: str.join('', zh_pattern.findall(x)))
    df[['text']].to_csv(file_path, index=False, header=False)


def write_chunk(raw_df: pd.DataFrame, lower_bound: int, upper_bound: int, file_count: int,
                file_type: str = 'csv') -> None:
    if upper_bound == lower_bound:
        raise ValueError('lower_bound = upper_bound = {}, key = {}'.format(lower_bound, raw_df['key'][lower_bound]))
    file_path = '{}/{}.{}.{}'.format(SUB_DIR_NAME, FILE_PREFIX, file_count, file_type)
    if file_type == 'csv':
        raw_df[lower_bound: upper_bound].to_csv(file_path, index=False)
    else:
        df_to_txt(raw_df[lower_bound: upper_bound], file_path)


def check_size(raw_df: pd.DataFrame, lower_bound: int, upper_bound: int, file_type: str = 'csv') -> int:
    if file_type == 'csv':
        raw_df[lower_bound: upper_bound].to_csv(BUF_FILE_NAME, index=False)
    else:
        df_to_txt(raw_df[lower_bound: upper_bound], BUF_FILE_NAME)

    result = os.path.getsize(BUF_FILE_NAME)
    os.remove(BUF_FILE_NAME)

    return result


def get_bound(df: pd.DataFrame, lower_bound: int, length: int, chunk_size: int, file_type: str = 'csv') -> int:
    left, right, ans = lower_bound, length, lower_bound
    while left <= right:
        mid = left + right >> 1
        if check_size(df, lower_bound, mid, file_type=file_type) <= chunk_size * 1024:
            ans = mid
            left = mid + 1
        else:
            right = mid - 1

    return ans


def split_df(df: pd.DataFrame, chunk_size: int = 900, file_type: str = 'csv') -> None:
    """
    将 df 分解为若干个小于 file_size 的 chunk
    :param df: DataFrame
    :param chunk_size: size in KB
    :param file_type output file type, csv or txt
    :return: None
    """

    length = len(df)

    file_count = 0
    current_index = 0

    while current_index < length:
        upper_bound = get_bound(df, current_index, length, chunk_size, file_type=file_type)
        write_chunk(df, current_index, upper_bound, file_count, file_type=file_type)
        current_index = upper_bound
        file_count += 1


def main():
    df = pd.read_csv(SOURCE_FILE_NAME)

    if not os.path.exists(SUB_DIR_NAME):
        os.mkdir(SUB_DIR_NAME)

    split_df(df, file_type='txt')


if __name__ == '__main__':
    os.chdir('../dataset')

    main()
