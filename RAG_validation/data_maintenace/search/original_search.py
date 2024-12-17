from utils.calc import calc_cos_similarity
from search.keyword_search import keyword_search
import sys
import os
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "utils")))
from request_llm import request_openai_embedding


def original_search(search_text: str, df: pd.DataFrame) -> list[str]:
    """
    Description: テキスト全体に対してハイブリッド検索を行う関数
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
    copy_df["keyword_rank"] = 0

    keyword_top = keyword_search(search_text, df)
    for index, keyword in enumerate(keyword_top):
        for i in range(len(copy_df)):
            sentence = copy_df["text"][i]
            if keyword == sentence:
                copy_df["keyword_rank"][i] = float(index + 1.0)

    copy_df["hybrid_score"] = 1 / (copy_df["vector_rank"] + 60) + 1 / (
        copy_df["keyword_rank"] + 60
    )

    copy_df["hybrid_rank"] = copy_df["hybrid_score"].rank(method="min", ascending=False)
    copy_df.to_excel(f"archive/hybrid_{search_text}.xlsx")
    top_df = copy_df.nsmallest(5, "hybrid_rank")
    return copy_df, top_df["text"].tolist()
