from func import chat_fine_tuning_model, check_fine_tunig_info


result = check_fine_tunig_info("ftjob-LjN0gg09gXsUJTno04KytyOw")
model_name = result.fine_tuned_model
user_prompt = "最近、寒くて布団から出られない。どうしよう？"
system_prompt = ""

tuning_model_message = chat_fine_tuning_model(model_name, system_prompt, user_prompt)
print(tuning_model_message)
