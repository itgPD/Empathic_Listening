import pandas as pd
from utils.big_query import BigQuery
from utils.request_llm import request_openai_chat, request_openai_embedding
import json
from search.hybrid_search import hybrid_search
from search.original_search import original_search
from search.original_vector_search import original_vector_search
from search.vector_search import vector_search
import const
import prompt
from utils.calc import calc_cos_similarity

big_query = BigQuery(const.PROJECT_ID, const.DATASET_ID)
df_1000 = big_query.get_table_as_df("label_dataset_1000")


search_text = (
    "リーンキャンバスにおける独自の価値提案を書く際のポイントについて教えてください"
)
text = original_vector_search(search_text, df_1000)
content = prompt.answer_prompt.format(
    question=search_text, text=text, format=prompt.answer_format
)
response = request_openai_chat(prompt.system_prompt, content)
print(response["answer"])
