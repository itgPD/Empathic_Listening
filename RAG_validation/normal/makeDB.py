# ベクトルDBを作成すための実行ファイル

from langchain.text_splitter import CharacterTextSplitter
import os
from dotenv import load_dotenv
from utils.request_llm import request_openai_embedding
import pandas as pd
from utils.big_query import BigQuery
import const
from tqdm import tqdm

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
folder_path = "./resource_text"
txt_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
chunk_docs = []
big_query = BigQuery(const.PROJECT_ID, const.DATASET_ID)
chunk_size = 1000

for file_name in txt_files:
    file_path = os.path.join(folder_path, file_name)
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
        lines = content.split("\n\n")
        for line in lines:
            chunk_docs.append(line)
            if len(line) < chunk_size:
                chunk_docs.append(line)
            else:
                text_splitter = CharacterTextSplitter(
                    separator="。", chunk_size=chunk_size, chunk_overlap=100
                )
                docs = text_splitter.split_text(content)
                chunk_docs.extend(docs)

df = pd.DataFrame(columns=["text", "vector", "id"], index=range(len(chunk_docs)))
for index, chunk_text in tqdm(enumerate(chunk_docs), total=len(chunk_docs)):
    df["text"][index] = chunk_text
    vector = request_openai_embedding(chunk_text)
    df["vector"][index] = vector
    df["id"][index] = index
