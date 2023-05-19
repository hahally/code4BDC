import csv
import random
import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,BertTokenizer
import torch
from scripts.dataset import create_BART_dataset
from scripts.utils import setup_seed
from tqdm import tqdm
'''
本文件是一个基本的demo，随机生成N条10-999之间的数据作为伪结果文件。
N 应该和标准结果中要求的数据行数相同。
'''
os.environ["WANDB_DISABLED"] = "true"
BATCH_SIZE = 32
SRC_MAX_LEN = 200
TGT_MAX_LEN = 90
SEED = 2222222
setup_seed(SEED)

def invoke(input_data_path, output_data_path):
    # 从输入地址 'input_data_path'读入待测试数据
    tokenizer = BertTokenizer.from_pretrained("./tokenizer/vocab.txt")
    model = AutoModelForSeq2SeqLM.from_pretrained("./pretrain_model/bart-base")
    model.resize_token_embeddings(len(tokenizer))
    model.load_state_dict(torch.load('./best_model.pkl')['model'])
    df = pd.read_csv(input_data_path, encoding='utf-8', header=None)
    test_loader = create_BART_dataset(src=df[1].tolist(),
                                    tgt=df[1].tolist(),
                                    tokenizer=tokenizer,
                                    src_max_len=SRC_MAX_LEN,
                                    tgt_max_len=TGT_MAX_LEN,
                                    batch_size=BATCH_SIZE,
                                    shuffle=False,
                                    pin_memory=False,
                                    num_workers=0)   

    # 生成预定义数据样本
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    predictions = []
    with torch.no_grad():
        for data in tqdm(test_loader):
            y = data['target_ids'].to(device, dtype = torch.long)
            ids = data['source_ids'].to(device, dtype = torch.long)
            mask = data['source_mask'].to(device, dtype = torch.long)
            generated_ids = model.generate(
                input_ids = ids,
                attention_mask = mask,
                max_length=90,
                num_beams=5,
                temperature=0.9,
                length_penalty=0.9,
                top_k=100,
                top_p=0.99
                )
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            predictions.extend(preds)

    # 将选手程序生成的结果写入目的地地址文件中
    df[1] = predictions
    df[[0,1]].to_csv(output_data_path, header=None, index=False)
    

if __name__ == '__main__':
    # invoke(input_data_path='./preliminary_b_test.csv',output_data_path='./pre.csv')
    pass