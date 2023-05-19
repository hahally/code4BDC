from transformers import (AutoTokenizer, 
                          AutoModelForMaskedLM, 
                          LineByLineTextDataset, 
                          DataCollatorForLanguageModeling, 
                          Trainer, 
                          TrainingArguments,
                          BertTokenizer)
import os
from torch.utils.data.dataloader import Dataset
import torch

os.environ["WANDB_DISABLED"] = "true"
# 加载tokenizer和模型
tokenizer = BertTokenizer.from_pretrained("./tokenizer")
model = AutoModelForMaskedLM.from_pretrained("./pretrain_model/bart-base/")
model.resize_token_embeddings(len(tokenizer))

model.config.bos_token_id = tokenizer.cls_token_id
model.config.eos_token_id = tokenizer.sep_token_id
model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.forced_eos_token_id = tokenizer.sep_token_id

class LineByLineTextDataset(Dataset):
    def __init__(self, tokenizer, file_path, block_size):
        if os.path.isfile(file_path) is False:
            raise ValueError(f"Input file path {file_path} not found")
        print(f"Creating features from dataset file at {file_path}")

        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
        self.examples = lines
        self.block_size = block_size
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        lines = [self.examples[i]]
        batch_encoding = self.tokenizer(lines, add_special_tokens=True, truncation=True, max_length=self.block_size)
        x = batch_encoding["input_ids"][0]
        
        return {"input_ids": torch.tensor(x, dtype=torch.long)}

# 加载要预训练的文本数据
dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="./data/corpus.txt",
    block_size=128
)
# 定义用于训练的数据收集器
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.3
)
# 定义训练参数
training_args = TrainingArguments(
    output_dir="./pretrain_model/bart-base-output",
    overwrite_output_dir=True,
    num_train_epochs=50,
    per_device_train_batch_size=32,
    gradient_accumulation_steps=4,
    save_steps=500,
    save_total_limit=2,
    prediction_loss_only=True,
    learning_rate=1e-4,
    weight_decay=0.01,
    warmup_steps=int(36000*0.1)
)
# 定义Trainer对象并开始训练
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset
)
trainer.train()