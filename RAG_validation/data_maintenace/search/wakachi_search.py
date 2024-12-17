import sys
import os
import pandas as pd

from search.vector_search import vector_search

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "utils")))
from request_llm import request_openai_chat


system_prompt = """
    これから依頼者である私があなたに対して質問をしますので、適切なフォーマットと制約を守って、JSON形式で返事をしてください。
"""

wakachi_prompt = """
    # 依頼
    これから# 文章をお渡しするので、以下の# 例を参考にして名詞の要素を全て抜き出してください。
    ただし、〇〇の〇〇は1つの名詞として抜き出してください。
    
    # 制約
    * 「違い」や「コツ」というワードは抜き出さないこと
    
    # 例
    文章: 学校の宿題はやらないといけない
    抜き出したい要素 : 学校の宿題
    
    # 例
    文章: 日本の政治と宗教について教えて
    抜き出したい要素 : 日本の政治、日本の宗教
    
    # 例
    文章: 学校の補修と補講の違いは何ですか？
    抜き出したい要素: 学校の補修、学校の補講

    
    # 文章
    {sentence}
    
    # フォーマット
    {formated}
"""

wakachi_format = """
        {
            noun:[xx,]
        }
"""


def wakachi_search(search_text: str, df: pd.DataFrame) -> list[str]:
    content = wakachi_prompt.format(sentence=search_text, formated=wakachi_format)
    response = request_openai_chat(system_prompt, content)
    response_dict = dict(response)
    noun_list = response_dict["noun"]
    print(noun_list)
    text_list = []
    for noun in noun_list:
        # copy_df, text = hybrid_search(noun, df)
        copy_df, text = vector_search(noun, df)
        copy_df.to_csv(f"archive/wakachi_{noun}.csv")
        text_list.extend(text)
    text_list = list(set(text_list))
    return text_list
