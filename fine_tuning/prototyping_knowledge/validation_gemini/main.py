# Gemini版のFine-tuningを行う実行ファイル

import time
import vertexai
from vertexai.preview.tuning import sft

vertexai.init(project="", location="us-central1")

sft_tuning_job = sft.train(
    source_model="gemini-1.0-pro-002",
    train_dataset="gs://2024_summer_intern/kabeuchi_data_gemini.jsonl",
    epochs=4,
    adapter_size=8,
    learning_rate_multiplier=1,
    tuned_model_display_name="tuned_test",
)

while not sft_tuning_job.has_ended:
    time.sleep(60)
    sft_tuning_job.refresh()

print(sft_tuning_job.tuned_model_name)
print(sft_tuning_job.tuned_model_endpoint_name)
print(sft_tuning_job.experiment)
