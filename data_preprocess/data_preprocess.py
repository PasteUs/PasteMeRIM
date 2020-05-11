"""
数据预处理
"""

import jieba
import re
import pandas as pd
import numpy as np
from zhon import hanzi


def tokenize(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    使用 jieba 分词
    :param raw_df: source DataFrame
    :return: df with column named "tokens"
    """

    df = raw_df.copy()
    zh_pattern = re.compile('[{}]'.format(hanzi.characters))
    df['tokens'] = df['content'].apply(
        lambda x: list(jieba.cut(str.join('', zh_pattern.findall(x))))
    )

    return df

