import pandas as pd
from langchain.text_splitter import CharacterTextSplitter


df = pd.read_excel("data/df_raw.xlsx")[["text", "vector", "id", "flg"]]


def split_sentence(df: pd.DataFrame) -> dict[str, str]:
    try:
        text_list = df["text"].tolist()
        cleaned_list = [item.replace("\n", "") for item in text_list]
        text = ",".join(cleaned_list)

        chunk_size = 500
        chunk_overlap = 200

        text_splitter = CharacterTextSplitter(
            separator="。", chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        docs = text_splitter.split_text(text)
        # print(len(docs))

        chunk_df = pd.DataFrame([{"text": []}])

        for idx, chunk_text in enumerate(docs):
            chunk_df.at[idx, "text"] = chunk_text

        chunk_df.to_excel(f"data/chunk_text_{chunk_size}.xlsx", index=False)
        return {"status": "sucsess"}
    except Exception as e:
        return {"status": "sucsess", "error": e}
