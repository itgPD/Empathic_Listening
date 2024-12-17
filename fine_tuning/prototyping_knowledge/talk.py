from func import (
    chat_fine_tuning_model,
    check_fine_tunig_info,
)

result = check_fine_tunig_info("ftjob-rYl7ivORoHFK9WxINGfRytQ2")

model_name = result.fine_tuned_model
user_prompt = "リーンキャンバスの課題を記入する際に重視することは？"
system_prompt = ""


tuning_model_message = chat_fine_tuning_model(model_name, system_prompt, user_prompt)
print(tuning_model_message)
