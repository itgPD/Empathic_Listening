import numpy as np
from utils.request_llm import request_openai_embedding, request_openai_chat
import prompt
from tqdm import tqdm
import pandas as pd
from janome.tokenizer import Tokenizer
from rank_bm25 import BM25Okapi
from typing import Any


def calc_cos_similarity(list1: list[float], list2: list[float]) -> float:
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


def vector_search(search_text: str, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Description: ベクトル検索を行う関数
    Input:
        search_text:検索にかけたい文章
        df: ベクトルDBに格納されているdf
    Output:
        ベクトル検索の結果が付与されたデータフレーム、ベクトル検索で上位3位に入った文章のテキスト
    """
    copy_df = df.copy()
    search_text_vector = request_openai_embedding(search_text)
    copy_df["cos_simi"] = copy_df["vector"].apply(
        lambda x: calc_cos_similarity(search_text_vector, x)
    )
    copy_df["vector_rank"] = copy_df["cos_simi"].rank(method="min", ascending=False)
    top_df = copy_df.nsmallest(3, "vector_rank")
    return copy_df, top_df["text"].tolist()


def split_keyword(search_text: str, df: pd.DataFrame) -> Any:
    """
    Description: キーワード検索を行う関数
    Input:
        search_text:検索にかけたい文章
        df: ベクトルDBに格納されているdf
    Output:
    """
    texts = df["text"].tolist()
    t = Tokenizer()

    def tokenize(text):
        return [token.surface for token in t.tokenize(text)]

    def query_tokenize(text):
        return [
            token.surface
            for token in t.tokenize(text)
            if token.part_of_speech.split(",")[0] in ["名詞", "形容詞"]
        ]

    tokenized_documents = [tokenize(doc) for doc in texts]
    bm25 = BM25Okapi(tokenized_documents)
    tokenized_query = query_tokenize(search_text)
    keyword_top = bm25.get_top_n(tokenized_query, texts, n=len(df))
    return keyword_top


def hybrid_search(search_text: str, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Description: ハイブリッド検索を行う関数
    Input:
        search_text:検索にかけたい文章
        df: ベクトルDBに格納されているdf
    Output:
        ハイブリッドの結果が付与されたデータフレーム、ハイブリッドで上位3位に入った文章のテキスト
    """
    copy_df = df.copy()
    search_text_vector = request_openai_embedding(search_text)
    copy_df["cos_simi"] = copy_df["vector"].apply(
        lambda x: calc_cos_similarity(search_text_vector, x)
    )
    copy_df["vector_rank"] = copy_df["cos_simi"].rank(method="min", ascending=False)
    copy_df["keyword_rank"] = 0

    keyword_top = split_keyword(search_text, df)
    for index, keyword in enumerate(keyword_top):
        for i in range(len(copy_df)):
            sentence = copy_df["text"][i]
            if keyword == sentence:
                copy_df["keyword_rank"][i] = float(index + 1.0)
    # ベクトル検索、キーワード検索の結果をランク付けしている
    copy_df["hybrid_score"] = 1 / (copy_df["vector_rank"] + 60) + 1 / (
        copy_df["keyword_rank"] + 60
    )
    copy_df["hybrid_rank"] = copy_df["hybrid_score"].rank(method="min", ascending=False)
    top_df = copy_df.nsmallest(3, "hybrid_rank")
    return copy_df.reset_index(drop=True), top_df["text"].tolist()


def subquery_search(search_text: str, df: pd.DataFrame) -> list[str]:
    """
    Description: サブクエリを生成し、ハイブリッド検索を行う関数
    Input:
        search_text:検索にかけたい文章
        df: ベクトルDBに格納されているdf
    Output:
        サブクエリごとの類似度が高い文章
    """
    content = prompt.subquery_prompt.format(
        question=search_text, format=prompt.subquery_format
    )
    subquery = request_openai_chat(prompt.system_prompt, content)
    subquery = list(subquery.values())
    text_list = []
    for query in subquery:
        _, text = hybrid_search(query, df)
        text_list.extend(text)
    text_list = list(set(text_list))
    return text_list


def wakachi_search_hybrid(search_text: str, df: pd.DataFrame) -> list[str]:
    """
    Description: 分かち書きで名詞句に絞り、ハイブリッド検索を行う関数
    Input:
        search_text:検索にかけたい文章
        df: ベクトルDBに格納されているdf
    Output:
        名詞句ごとの類似度が高い文章
    """
    content = prompt.wakachi_prompt.format(
        sentence=search_text, formated=prompt.wakachi_format
    )
    response = request_openai_chat(prompt.system_prompt, content)
    response_dict = dict(response)
    noun_list = response_dict["noun"]
    text_list = []
    for noun in noun_list:
        _, text = hybrid_search(noun, df)
        text_list.extend(text)
    text_list = list(set(text_list))
    return text_list


def search_RAGFUSION(search_text: str, df: pd.DataFrame) -> dict[str, str]:
    """
    Description: RAG Fusionを用いて回答を生成する関数
    Input:
        search_text:検索にかけたい文章
        df: ベクトルDBに格納されているdf
    Output:
        RAG_Fusionで生成された回答
    """
    content = prompt.subquery_prompt.format(
        question=search_text, format=prompt.subquery_format
    )
    # 類似クエリの生成
    subquery = request_openai_chat(prompt.system_prompt, content)
    subquery = list(subquery.values())
    ans_list = []
    for query in subquery:
        cos_simirality = []
        query_vector = request_openai_embedding(query)
        for idx in range(len(df)):
            vector = df["vector"][idx]
            calc_simi = calc_cos_similarity(query_vector, vector)
            cos_simirality.append(calc_simi)
        top_index = sorted(
            range(len(cos_simirality)),
            key=lambda i: cos_simirality[i] - 1,
            reverse=True,
        )[:3]
        filter_df = df.iloc[top_index].reset_index(drop=True)
        text = filter_df["text"].tolist()

        # 各クエリでの回答を生成する
        content = prompt.answer_prompt.format(
            question=search_text, text=text, format=prompt.answer_format
        )
        response = request_openai_chat(prompt.system_prompt, content)
        answer = response["answer"]
        ans_list.append(answer)

    # 最後に各クエリの回答をまとめて、最終的な回答を生成さえる
    content = prompt.rag_fusion_prompt.format(
        quesion=search_text, answer=ans_list, format=prompt.rag_fusion_format
    )
    response = request_openai_chat(prompt.system_prompt, content)
    return response["answer"]
