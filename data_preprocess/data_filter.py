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


def main():
    df = pd.read_csv('permanents.csv')

    df = df[df['key'] != 12643]
    df = df[df['key'] != 12648]

    df[get_filter_column(df)].to_csv('filtered_permanent.csv', index=False)


if __name__ == '__main__':
    os.chdir('../dataset')

    main()
