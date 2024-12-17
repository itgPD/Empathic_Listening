from utils.calc import calc_cos_similarity
import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "utils")))
from request_llm import request_openai_embedding


def vector_search(search_text: str, df: pd.DataFrame) -> list[str]:
    """
    Description: テキストのラベルに対してベクトル検索を行う関数
    Input:
        search_text: 検索したい文章
        df: ベクトルDBに格納されているdf
    Output:
        類似度が高い上位3つの文章

    """
    copy_df = df.copy()
    search_text_vector = request_openai_embedding(search_text)
    copy_df["cos_simi"] = copy_df["summary_vector"].apply(
        lambda x: calc_cos_similarity(search_text_vector, x)
    )
    copy_df["vector_rank"] = copy_df["cos_simi"].rank(method="min", ascending=False)
    copy_df["reverse_cos_simi"] = copy_df["reverse_summary_vector"].apply(
        lambda x: calc_cos_similarity(search_text_vector, x)
    )
    copy_df["reverse_vector_rank"] = copy_df["reverse_cos_simi"].rank(
        method="min", ascending=False
    )

    copy_df["multi_score"] = 1 / (copy_df["vector_rank"] + 60) + 1 / (
        copy_df["reverse_vector_rank"] + 60
    )
    copy_df["multi_rank"] = copy_df["multi_score"].rank(method="min", ascending=False)
    top_df = copy_df.nsmallest(5, "multi_rank")
    return top_df["text"].tolist()
