from utils.big_query import BigQuery
import const
from utils.search_document import (
    subquery_search,
    wakachi_search_hybrid,
    vector_search,
    hybrid_search,
    search_RAGFUSION,
)
from utils.request_llm import request_openai_chat
import prompt

big_query = BigQuery(const.PROJECT_ID, const.DATASET_ID)
df = big_query.get_table_as_df(const.data_paragraph)

search_text = "顧客インタビューとはなんですか？そもそもどんな目的でどのようにやるのか教えてください。"
text = vector_search(search_text, df)
content = prompt.answer_prompt.format(
    question=search_text, text=text, format=prompt.answer_format
)
response = request_openai_chat(prompt.system_prompt, content)
print(response)
