import numpy as np


def calc_cos_similarity(list1: list[float], list2: list[float]) -> list[float]:
    """
    Description: cos類似度を計算する関数
    Input:
        list1, list2: ベクトルのリスト
    Output:
        ベクトルのcos類似度
    """
    vector1 = np.array(list1)
    vector2 = np.array(list2)
    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)
    return dot_product / (magnitude1 * magnitude2)
