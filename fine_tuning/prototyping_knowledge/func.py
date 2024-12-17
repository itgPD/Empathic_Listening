import os
from openai import OpenAI
from dotenv import load_dotenv
import vertexai
from vertexai.preview.tuning import sft
from vertexai.preview.generative_models import GenerativeModel
from vertexai.preview import tuning
from typing import Union, Optional
from type.type import FineTuningJob

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)


def upload_data(path: str) -> str:
    """
    Description: 訓練用のデータをアップロードするための関数
    Input:
        path: 訓練データがあるフォルダ名
    Output:
        file_id: OpneAIにアップロードされた訓練データのファイルID
    """
    response = client.files.create(file=open(path, "rb"), purpose="fine-tune")
    print("アップロード完了")
    file_id = response.id
    return file_id


def execute_fine_tuning(
    file_id: str,
    epoch: Union[int, float],
    batch_size: int,
    learning_rate: Union[int, float],
    validation_id: Optional[str],
) -> str:
    """
    Description: Fine-tuningを実行するための関数
    Input:
        file_id: OpneAIにアップロードされた訓練データのファイルID
        epoch: エポック数
        batch_size: バッチサイズ
        learning_rate: 学習率
        validation_id: 検証用データ
    Output:
        job_id: 実行されているFine-tuningのjob_id

    """

    if validation_id:
        response_job = client.fine_tuning.jobs.create(
            training_file=file_id,
            validation_file=validation_id,
            model="gpt-4o-2024-08-06",
            hyperparameters={
                "n_epochs": epoch,
                "batch_size": batch_size,
                "learning_rate_multiplier": learning_rate,
            },
        )
        job_id = response_job.id
        return job_id
    else:
        response_job = client.fine_tuning.jobs.create(
            training_file=file_id,
            model="gpt-4o-2024-08-06",
            hyperparameters={
                "n_epochs": epoch,
                "batch_size": batch_size,
                "learning_rate_multiplier": learning_rate,
            },
        )
        job_id = response_job.id
        return job_id


def check_fine_tunig_info(job_id: str) -> FineTuningJob:
    """
    Description: Fine-tuningの実行状況を確認する関数
    Input:
        job_id: 実行したFine-tuningのjob_id
    Ouput:
        Fine-tuningの実況状況
    """
    response_retrieve = client.fine_tuning.jobs.retrieve(job_id)
    return response_retrieve


def cancel_fine_tuning(job_id: str) -> dict[str, Union[str, Exception]]:
    """
    Description: Fine-tuningの実行をキャンセルする関数
    Input:
        job_id: キャンセルFine-tuningのjob_id
    Output:
        キャンセルの実行情報
    """
    try:
        client.fine_tuning.jobs.cancel(job_id)
        return {"info": "キャンセル済み"}
    except Exception as e:
        return {"error": e}


def delete_model(model_name: str) -> dict[str, Union[str, Exception]]:
    """
    Description: Fine-tuning済みモデルを削除するための関数
    Input:
        Fine-tunig済みのモデル名
    Output:
        削除の実行情報
    """
    try:
        client.models.delete(model_name)
        return {"info": "削除済み"}
    except Exception as e:
        return {"error": e}


def chat_fine_tuning_model(
    model_name: str, system_prompt: str, user_prompt: str
) -> str:
    """
    Description: Fibne-tuningしたモデルと会話するための関数
    Input:
        model_name: Fine-tunig済みのモデル名
        system_prompt: システムプロンプト
        user_prompt: プロンプト
    """
    response = client.chat.completions.create(
        model=model_name,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
    )
    return response.choices[0].message.content


def chat_vertexai_tuning_model(resource_id: str, content: str) -> str:
    """
    Description: GeminiでFine-tuning済みのモデルで会話するための関数
    Input:
        resource_id:リソース名（"projects/783872951114/locations/us-central1/tuningJobs/8724043227830681600"）
        content: 会話したい内容
    Output:
        会話の応答内容
    """
    sft_tuning_job = sft.SupervisedTuningJob(resource_id)
    tuned_model = GenerativeModel(sft_tuning_job.tuned_model_endpoint_name)
    response = tuned_model.generate_content(content)
    return response.candidates[0].content.parts[0].text
