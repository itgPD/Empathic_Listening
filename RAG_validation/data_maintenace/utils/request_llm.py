import os
from dotenv import load_dotenv
from openai import OpenAI
import const as const
import json
import numpy as np
from numpy.typing import NDArray

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


def request_openai_chat(system: str, content: str) -> dict[str, str]:
    """
    Description: GPTのchatエンドポイントにアクセス関数
    Input:
        system: システムプロンプト
        content: プロンプト
    Output
        プロンプトに対する応答
    """
    client = OpenAI(api_key=openai_api_key)
    if system:
        response = client.chat.completions.create(
            model=const.OPENAI_GPT_MODEL_VER_4O,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        response_message = response.choices[0].message.content
        return json.loads(response_message)
    else:
        response = client.chat.completions.create(
            model=const.OPENAI_GPT_MODEL_VER_4O,
            messages=[
                {"role": "user", "content": content},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        response_message = response.choices[0].message.content
        return json.loads(response_message)


def request_openai_embedding(text: str) -> NDArray[np.float64]:
    """
    Description: GPTのembeddingエンドポイントにアクセスする関数
    Input:
        text: ベクトル化したいテキスト
    Output:
        テキストのベクトル
    """
    client = OpenAI(api_key=openai_api_key)
    response = client.embeddings.create(model=const.EMBEDDING_OPENAI_SMALL, input=text)
    return np.array(response.data[0].embedding)
