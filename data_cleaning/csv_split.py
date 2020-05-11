"""
doccano 最大只能上传 1MB 的文件，需要将 CSV 拆分
"""

import os
import pandas as pd

SUB_DIR_NAME = 'batch_filtered_permanent'
FILE_PREFIX = 'filtered_permanent'
BUF_FILE_NAME = '.buf.csv'
SOURCE_FILE_NAME = 'filtered_permanent.csv'


def write_chunk(df: pd.DataFrame, lower_bound: int, upper_bound: int, file_count: int) -> None:
    if upper_bound == lower_bound:
        raise ValueError('lower_bound = upper_bound = {}, key = {}'.format(lower_bound, df['key'][lower_bound]))
    file_path = '{}/{}.{}.csv'.format(SUB_DIR_NAME, FILE_PREFIX, file_count)
    df[lower_bound: upper_bound].to_csv(file_path, index=False)


def check_size(df: pd.DataFrame, lower_bound: int, upper_bound: int) -> int:
    df[lower_bound: upper_bound].to_csv(BUF_FILE_NAME, index=False)
    result = os.path.getsize(BUF_FILE_NAME)
    os.remove(BUF_FILE_NAME)
    return result


def get_bound(df: pd.DataFrame, lower_bound: int, length: int, chunk_size: int) -> int:
    left, right, ans = lower_bound, length, lower_bound
    while left <= right:
        mid = left + right >> 1
        if check_size(df, lower_bound, mid) <= chunk_size * 1024:
            ans = mid
            left = mid + 1
        else:
            right = mid - 1

    return ans


def split_df(df: pd.DataFrame, chunk_size: int = 768) -> None:
    """
    将 df 分解为若干个小于 file_size 的 chunk
    :param df: DataFrame
    :param chunk_size: size in KB
    :return: None
    """

    length = len(df)

    file_count = 0
    current_index = 0

    while current_index < length:
        upper_bound = get_bound(df, current_index, length, chunk_size)
        write_chunk(df, current_index, upper_bound, file_count)
        current_index = upper_bound
        file_count += 1


def main():
    df = pd.read_csv(SOURCE_FILE_NAME)

    if not os.path.exists(SUB_DIR_NAME):
        os.mkdir(SUB_DIR_NAME)

    split_df(df)


if __name__ == '__main__':
    os.chdir('../dataset')

    main()
