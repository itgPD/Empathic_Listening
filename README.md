# 「プロトタイピング企画壁打ちくん」

## フォルダ構成

大きく分けて、RAG 用と Fine-Tuning 用の 2 つのフォルダが存在します。  
これらのフォルダにはあくまでも実装例を載せているだけなので、個人で実行環境を用意するようにしてください。

```
├── RAG*validation
│ ├── data_maintenace
│ └── normal
│
├── fine_tuning
│ ├── prototyping_knowledge
│ └── turtle_talk

```

### フォルダの説明

- Fine-Tuning  
  こちらのフォルダは大きく、キャラ付け検証用（turtle_talk）とプロトタイピングの知識植え付け（prototyping_knowledge）の 2 つに分かれています。

  - turtle_talk

    ```
      ├── data
      │   ├── turtle_talk_test_data.jsonl
      │   └── turtle_talkdatajsonl
      │
      ├── func.py //Fine-Tuningにおける関数をまとめたファイル
      ├── main.py //Fine-Tuning実行
      ├── talk.py //Fine-Tuning済みモデルと
      │
      └── type
          └── type.py //Fine-Tuningのレスポンスの型を定義したファイル
    ```

  - prototyping_knowledge

    ```
        ├── data
        │   ├── kabeuchi_data.jsonl
        │   ├── kabeuchi_test_data.jsonl
        │   └── prototyping_knowledge.txt
        │
        ├── func.py  //Fine-Tuningにおける関数をまとめたファイル
        ├── main.py //Fine-Tuning実行
        ├── talk.py //Fine-Tuning済みモデルと
        │
        ├── type
        │   └── type.py //Fine-Tuningのレスポンスの型を定義したファイル
        │
        └── validation_gemini
            └── main.py //Gemini版でのFine-Tuning
    ```

- RAG_validation  
  こちらのフォルダも大きく、通常の RAG や検索方法につい手の検証用（normal）とデータ整備に焦点を当てた RAG の検証（data_data_maintenace）の 2 つに分かれています。

  - normal

    ```
      ├── const.py //モデル・プロジェクトIDなどをまとめた定数ファイル
      ├── main.py //RAGの実行ファイル
      ├── makeDB.py //ベクトルデータベースを作成するファイル
      ├── prompt.py //プロンプトをまとめたファイル
      ├── resource_text //文書として使用しているテキストファイル
      |
      └── utils
          ├── big_query.py //bigqueryに関してまとめたファイル
          ├── request_llm.py //LLMのAPIをまとめたファイル
          └── search_document.py //文書の検索についてまとめたファイル
    ```

  - data_maintenace

    ```
      ├── chunk_text.py //テキストをチャンクサイズに分割するファイル
      ├── const.py //モデル・プロジェクトIDなどをまとめた定数ファイル
      ├── labeling.py //チャンクごとのテキストに対して、labelを付与するファイル
      ├── main.py //RAGの実行ファイル
      ├── prompt.py //プロンプトをまとめたファイル
      |
      ├── search
      │   ├── hybrid_search.py
      │   ├── original_search.py
      │   ├── original_vector_search.py
      │   ├── subquery_search.py
      │   ├── vector_search.py
      │   └── wakachi_search.py
      |
      └── utils
          ├── big_query.py //bigqueryに関してまとめたファイル
          ├── calc.py //計算についてまとめたファイル
          ├── request_llm.py //LLMのAPIをまとめたファイル
    ```

## 使い方

各フォルダごとの実行方法は以下の通りです。

- turtle_talk

  ```
  # Fine-Tuningを実行する
    python main.py

  # Fine-Tuningしたモデルを会話を行う
    python talk.py
  ```

  > **⚠ 注意**
  >
  > `python main.py` を実行すると、`job_id` が発行されるので、それを `talk.py` の `check_fine_tunig_info()` の引数に入れてください。  
  > OpenAI の API キーが必要になります。`.env`ファイルを作成して、OPENAI_API_KEY に値を記入してください。

  <br>

- prototyping_knowledge

  ```
  # Fine-Tuningを実行する
    python main.py

  # Fine-Tuningしたモデルを会話を行う
    python talk.py
  ```

  > **⚠ 注意**
  >
  > `python main.py` を実行すると、`job_id` が発行されるので、それを `talk.py` の `check_fine_tunig_info()` の引数に入れてください。

  <br>

- normal

  ```
  # 情報検索を行い、回答を生成する
    python main.py
  ```

  > **⚠ 注意**
  >
  > あくまでも実装例であり、そのまま実行することはできません。 　
  > `main.ipynb`も準備しているので、一連の流れを見たい方はそちらを参照ください  
  > 質問したい内容`search_text`と検索方法`vector_search, hybrid_search`などを変更する必要があります  
  > データを変更したい場合は、`resource_text`の中にテキストファイルを追加して、`makeDB.py`を実行してください

  <br>

- data_maintenace

  ```
  # 情報検索を行い、回答を生成する
    python main.py
  ```

  > **⚠ 注意**
  >
  > あくまでも実装例であり、そのまま実行することはできません。  
  > `main.ipynb`も準備しているので、一連の流れを見たい方はそちらを参照ください  
  > データを変更したい場合は、`resource_text`の中にテキストファイルを追加して、`labeling.py/chunk_sentence.py`を実行してください
