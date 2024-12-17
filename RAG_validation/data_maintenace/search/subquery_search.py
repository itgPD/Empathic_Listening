import sys
import os
import pandas as pd
from search.hybrid_search import hybrid_search
from search.vector_search import vector_search
import prompt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "utils")))
from request_llm import request_openai_chat


subquery_prompt = """
    # 依頼
    これからテキスト検索をする際に、質問の意図が伝わるようにしたいと考えています。
    そこで質問「{question}」に対して、以下の# 例を参考に文章を分解してください。
    
    # 例
    質問 : 低位安定とデジタル産業宣言の意味をそれぞれ教えて
    期待する回答 : 低位安定は何ですか？,デジタル産業宣言は何ですか？
    
    質問 : 学校の補修と補講の違いについて教えて
    期待する回答 : 学校の補修は何ですか？, 学校の補講は何ですか？
    
    # フォーマット
    {format}
"""

subquery_format = """
    {
        sentence1:xx,
    }
"""


def subquery_search(search_text: str, df: pd.DataFrame) -> list[str]:
    content = subquery_prompt.format(question=search_text, format=subquery_format)
    subquery = request_openai_chat(prompt.system_prompt, content)
    subquery = list(subquery.values())
    print(subquery)
    text_list = []
    for query in subquery:
        # copy_df, text = hybrid_search(query, df)
        copy_df, text = vector_search(query, df)
        copy_df.to_excel(f"archive/subquery_{query}.xlsx")
        text_list.extend(text)
    text_list = list(set(text_list))
    return text_list
