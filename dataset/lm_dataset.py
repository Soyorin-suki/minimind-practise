import json

from torch.utils.data import Dataset
import torch
import os
import random
from datasets import load_dataset
from transformers import PreTrainedTokenizer

# 禁用 HuggingFace tokenizer 的多进程并行，避免在 DataLoader 多进程环境中产生死锁
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class PretrainDataset(Dataset):

	def __init__(
		self,
		data_path: str,
		tokenizer: PreTrainedTokenizer,
		max_length: int = 512
	):
		super().__init__()
		self.tokenizer = tokenizer
		# 输入给GPU的最大长度
		self.max_length = max_length
		# 使用 HuggingFace 的惰性加载，防止一次性载入过多数据
		self.samples = load_dataset("json", data_files=data_path, split="train")

	def __len__(self):
		return len(self.samples)

	# 我们拿到的是 jsonl 里的每一行
	# tokenizer 把文本转化为
	def __getitem__(self, index):
		sample = self.samples[index]
		tokens = self.tokenizer(
			str(sample["text"]),
			add_special_token = False,
			max_length = self.max_length-2, 	# 给 BOS 和 EOS 留出位置
			truncation = True					# 如果长度超出了 max, 自动剪切
			).input_ids
		
		# 在首尾添加 BOS 和 EOS
		tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
		# 补全长度为 max_length
		input_ids = tokens + [self.tokenizer.pad_token_id]*(self.max_length - len(tokens))
		# 将其转为张量
		input_ids = torch.tensor(input_ids, dtype=torch.long)

		# labels 与 input_ids 完全相同，但是 PAD 位置设为 -100,
		# CrossEntroyLoss 会默认忽略 -100, 不计入 loss
		labels = input_ids.clone()
		labels[labels == self.tokenizer.pad_token_id] = -100
		attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

		return {
			"input_ids": input_ids,
			"attention_mask": attention_mask,
			"labels": labels
		}
	

		

