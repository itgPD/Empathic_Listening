from janome.tokenizer import Tokenizer
from rank_bm25 import BM25Okapi
import pandas as pd


def keyword_search(search_text: str, df: pd.DataFrame) -> pd.DataFrame:
    texts = df["text"].tolist()

    t = Tokenizer()

    def tokenize(text):
        return [token.surface for token in t.tokenize(text)]

    # クエリ用のTokenizerの定義
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
