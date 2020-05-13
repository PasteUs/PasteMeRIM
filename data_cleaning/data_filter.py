"""
过滤出疑似负样本
"""

import re
import os
import pandas as pd
from zhon import hanzi


def get_filter_column(raw_df: pd.DataFrame) -> pd.Series:
    zh_pattern = re.compile('[{}]'.format(hanzi.characters))
    code_pattern = re.compile(r'#include |void |import |def |int |return |for |for\(|{[\n\r]')
    html_pattern = re.compile(r'</\w+>')

    df = raw_df.copy()
    return df['content'].apply(
        lambda x: (code_pattern.search(x) is None) and
                  (html_pattern.search(x) is None) and
                  (zh_pattern.search(x) is not None)
    )


def erase_by_key(raw_df: pd.DataFrame, key_list: list) -> pd.DataFrame:
    df = raw_df.copy()

    for key in key_list:
        df = df[df['key'] != key]

    return df


def main():
    df = pd.read_csv('permanents.csv')

    df = erase_by_key(df, [12643, 12648, 25149, 25150])

    df = df[get_filter_column(df)]

    result_df = pd.DataFrame(columns=['label'])
    result_df['text'] = df['content']

    result_df[['text', 'label']].fillna('normal').to_csv('filtered_permanent.csv', index=False)


if __name__ == '__main__':
    os.chdir('../resources/dataset')

    main()
