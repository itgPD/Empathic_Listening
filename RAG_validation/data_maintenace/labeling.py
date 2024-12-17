import pandas as pd

from utils.request_llm import request_openai_chat, request_openai_embedding
import prompt
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.big_query import BigQuery
import const

df = pd.read_excel("data/chunk_text_500.xlsx")

text_list = df["text"].tolist()


def process_summary(idx: int, text: str) -> tuple[int, str]:
    """
    Description: テキストからサマリーを作成してもらう関数
    Input
        idx: インデックス
        text: インデックスに該当するテキスト
    Output
        idx: インデックス
        response["summary"]: サマリーテキスト
    """
    content = prompt.summary_prompt.format(data=text, format=prompt.summary_format)
    response = request_openai_chat(prompt.labeling_system_prompt, content)
    return idx, response["summary"]


with ThreadPoolExecutor() as executor:
    futures = {
        executor.submit(process_summary, idx, text): idx
        for idx, text in enumerate(text_list)
    }

    for future in tqdm(as_completed(futures), total=len(text_list)):
        idx, label = future.result()
        df.at[idx, "LLM_summary"] = label


def process_explain_text(idx: int, text: str) -> tuple[int, str]:
    """
    Description: サマリーのテキストから具体的な説明を補完させる関数
    Input
        idx: インデックス
        text: インデックスに該当するテキスト
    Output
        idx: インデックス
        response["summary"]:
    """
    content = prompt.reverse_title_prompt.format(
        text=text.text, content=text.LLM_summary, format=prompt.reverse_title_format
    )
    response = request_openai_chat(prompt.labeling_system_prompt, content)
    return idx, response["content"]


with ThreadPoolExecutor() as executor:
    futures = {
        executor.submit(process_explain_text, idx, text): idx
        for idx, text in enumerate(df.itertuples(index=False))
    }

    for future in tqdm(as_completed(futures), total=len(text_list)):
        idx, label = future.result()
        df.at[idx, "LLM_reveerse_summary"] = label


big_query = BigQuery(const.PROJECT_ID, const.DATASET_ID)
df["text_vector"] = ""
df["summary_vector"] = ""
df["reverse_summary_vector"] = ""


def process_embedding(idx: int) -> dict:
    """
    Description: テキストに対してembeddingを行うための関数
    Input
        idx: インデックス
    Output
        results: それぞれのテキストに対するvector
    """
    text = df["text"][idx]
    text_vector = request_openai_embedding(text)
    summary = df["LLM_summary"][idx]
    summary_vector = request_openai_embedding(summary)
    reverse_summary = df["LLM_reveerse_summary"][idx]
    reverse_summary_vector = request_openai_embedding(reverse_summary)
    results = {
        "idx": idx,
        "text_vector": text_vector,
        "summary_vector": summary_vector,
        "reverse_summary_vector": reverse_summary_vector,
    }

    return results


with ThreadPoolExecutor() as executor:
    futures = [executor.submit(process_embedding, idx) for idx in range(len(df))]
    for future in tqdm(futures):
        result = future.result()
        idx = result["idx"]
        df.at[idx, "text_vector"] = result["text_vector"]
        df.at[idx, "summary_vector"] = result["summary_vector"]
        df.at[idx, "reverse_summary_vector"] = result["reverse_summary_vector"]

big_query.upload_df(df, f"label_dataset_500")
