from func import (
    upload_data,
    execute_fine_tuning,
)

# 訓練データをアップロード
train_file_path = "data/talk.jsonl"
fild_id = upload_data(train_file_path)
print(f"アップロードされたファイル:{fild_id}")

epoch = 8
batch_size = 1
learning_rate = 2

job_id = execute_fine_tuning(fild_id, epoch, batch_size, learning_rate)
print(f"実行されているjob_id:{job_id}")