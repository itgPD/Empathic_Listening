import json
import pandas as pd
from google.cloud import bigquery

from google.cloud.exceptions import NotFound
from google.cloud.bigquery import LoadJobConfig


class BigQuery:
    """
    BigQuery関連の処理をまとめたクラス
    """

    def __init__(self, project_id, dataset_id):
        self.client = bigquery.Client(
            project=project_id,
        )
        self.project_id = project_id
        self.dataset_id = dataset_id

    def upload_df(self, df, table_id, write_disposition="WRITE_TRUNCATE"):
        """
        DataFrameをBQテーブルにアップロードする関数

        Args:
            df : dataframe
            table_id: string
                BQのテーブルID
            write_disposition: enum
                書き込みオプション
                ("WRITE_TRUNCATE"|"WRITE_APPEND"|"WRITE_EMPTY")

        Returns:
            None
        """
        table_ref = self.client.dataset(self.dataset_id).table(table_id)

        job_config = bigquery.LoadJobConfig()
        # job_config.source_format = bigquery.SourceFormat.CSV
        # job_config.skip_leading_rows = 1  # ヘッダー行をスキップ
        job_config.write_disposition = write_disposition
        job_config.autodetect = True

        job = self.client.load_table_from_dataframe(
            df, table_ref, job_config=job_config
        )
        job.result()
        print(f"Loaded {job.output_rows} rows to {self.dataset_id}:{table_id}.")

    def get_table_as_df(self, table_id) -> pd.DataFrame:
        """
        BQテーブル全件をDFとして取得する関数

        Args:
            table_id: string
                BQのテーブルID

        Returns:
            df
        """

        table = f"{self.project_id}.{self.dataset_id}.{table_id}"
        # rows = self.client.list_rows(table)
        # df = pd.DataFrame([dict(row.items()) for row in rows])

        query = f"SELECT * FROM {table}"
        query_job = self.client.query(query)  # APIリクエストを実行
        df = query_job.to_dataframe()
        return df

    def get_result_as_df(self, query) -> pd.DataFrame:
        """
        クエリの実行結果をDFとして取得する関数

        Args:
            query: string
                実行するクエリ

        Returns:
            df
        """

        df = self.client.query(query).to_dataframe()
        return df

    def insert_df(self, df, table_id):
        """
        DataFrameを既存のテーブルにinsertする関数

        Args:
            df: dataframe
            table_id: string

        Returns:
            job: Sequence[Sequence[dict]]
                JOB実行結果
        """
        table_ref = self.client.dataset(self.dataset_id).table(table_id)
        job = self.client.insert_rows_from_dataframe(
            self.client.get_table(table_ref), df
        )
        return job

    def load_df_to_exist_table(self, df, table_id, white_method="empty"):
        """
        DataFrameを既存のテーブルにloadする関数

        Args:
            df: dataframe
            table_id: string
            write_method: enum
                書き込みオプション
                ("truncate"|"append"|"empty")

        Returns:
            job: Sequence[Sequence[dict]]
                JOB実行結果

        Refference:
            https://rinoguchi.net/2019/12/bigquery-python.html
        """
        # 対象テーブルを取得
        table = self.client.get_table(f"{self.project_id}.{self.dataset_id}.{table_id}")

        # dfをjsonに変換
        json_rows = json.loads(df.to_json(orient="records", force_ascii=False))

        if white_method == "append":
            job_config = LoadJobConfig(
                schema=table.schema,
                write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
            )
        elif white_method == "truncate":
            job_config = LoadJobConfig(
                schema=table.schema,
                write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
            )
        else:
            job_config = LoadJobConfig(
                schema=table.schema,
                write_disposition=bigquery.WriteDisposition.WRITE_EMPTY,
            )

        # データロード
        job = self.client.load_table_from_json(json_rows, table, job_config=job_config)
        result = job.result()
        return result

    def delete_data(self, table_id):
        """
        BQテーブルからすべてのデータを削除する関数

        Args:
            table_id: string

        Returns:
            None
        """

        query = """
        DELETE FROM `{project_id}.{dataset_id}.{table_id}`
        WHERE 1=1
        """.format(
            project_id=self.project_id, dataset_id=self.dataset_id, table_id=table_id
        )
        job = self.client.query(query)
        return job.result()

    def create_empty_table(self, table_id_new, table_id_ref):
        """
        既存のテーブルのスキーマを参照して、空のテーブルを作成する

        Args:
            table_id_new: string
                新規作成するテーブルID
            table_id_ref: string
                参照するID
        Returns:
            None
        """

        table_ref = self.client.get_table(
            f"{self.project_id}.{self.dataset_id}.{table_id_ref}"
        )
        schema_ref = table_ref.schema

        table = bigquery.Table(
            f"{self.project_id}.{self.dataset_id}.{table_id_new}", schema=schema_ref
        )
        table = self.client.create_table(table)
        print(
            "Created table {}.{}.{}".format(
                table.project, table.dataset_id, table.table_id
            )
        )

    def table_exists(self, table_id):
        """
        テーブルの存在を確認する関数

        Args:
            table_id: string

        Returns:
            exist: boolean
        """

        table_ref = self.client.dataset(self.dataset_id).table(table_id)

        try:
            self.client.get_table(table_ref)
            return True

        except NotFound:
            return False
