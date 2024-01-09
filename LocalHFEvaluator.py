import os, torch
from LLMNeedleHaystackTester import LLMNeedleHaystackTester
from transformers import AutoModelForCausalLM, AutoTokenizer

class LocalHFEvaluator(LLMNeedleHaystackTester):
	def __init__(self, model_path, model_args={}, **kwargs):
		self.model_name = model_path
		self.model_to_test_description = model_path

		self.model, self.tokenizer = self.load_model_tokenizer(model_path, model_args)

		super().__init__(**kwargs)

	def load_model_tokenizer(self, model_path, model_args):
		model = AutoModelForCausalLM.from_pretrained(
			model_path,
			**model_args
		)
		tokenizer = AutoTokenizer.from_pretrained(model_path)

		return model, tokenizer 

	def get_encoding(self,context):
		return self.tokenizer.encode(context)

	def get_decoding(self, encoded_context):
		return self.tokenizer.decode(encoded_context)

	def get_prompt(self, context):
		return [
			# {
			# 	"role": "system",
			# 	"content": "You are a helpful AI bot that answers questions for a user. Keep your response short and direct"
			# },
			# {
			# 	"role": "user",
			# 	"content": context
			# },
			{
				"role": "user",
				"content": f"{context}\n\n{self.retrieval_question} Don't give information outside the document or repeat your findings"
			}
		]

	async def get_response_from_model(self, messages):
		input_tokenized = self.tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")

		with torch.no_grad():
			output_tokenized = self.model.generate(
				input_tokenized, 
				max_new_tokens=300, 
				do_sample=True)
		output_tokenized=output_tokenized[0][len(input_tokenized[0]):]
		output = self.tokenizer.decode(output_tokenized, skip_special_tokens=True)

		return output

if __name__ == "__main__":
	ht = LocalHFEvaluator(
		"models/Mistral-7B-Instruct-v0.2", 
		{
			"torch_dtype": torch.bfloat16, 
			"device_map": "auto",
	        "use_flash_attention_2": True
		},
		evaluation_method="not-gpt4", 
		context_lengths_max=10_000, 
		context_lengths_num_intervals=10, 
		document_depth_percent_intervals=10)

	ht.start_test()





