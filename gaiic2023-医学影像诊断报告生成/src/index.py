import os
import pandas as pd
from transformers import BartForConditionalGeneration,BertTokenizer
import torch
from scripts.dataset import create_BART_dataset
from scripts.utils import setup_seed
from scripts.models import BartForConditionalGenerationWithCopyMech
from tqdm import tqdm
from scripts.ensemble import Ensemble
'''
本文件是一个基本的demo，随机生成N条10-999之间的数据作为伪结果文件。
N 应该和标准结果中要求的数据行数相同。
'''
os.environ["WANDB_DISABLED"] = "true"
BATCH_SIZE = 32
SRC_MAX_LEN = 160
TGT_MAX_LEN = 80
SEED = 2222222
setup_seed(SEED)

model_path_list = ['./bart-base-ema-fgm','./bart-base-rdrop-2023','./bart-base-rdrop-6789','./bart-base-rdrop-seed567']
vocab_path = './bart-base-ema-fgm/'

def get_models():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_list = []
    for model_file in model_path_list:
        model = BartForConditionalGeneration.from_pretrained(model_file).to(device)
        model.eval()
        model_list.append(model)
        
    return model_list

def invoke(input_data_path, output_data_path):
    # 从输入地址 'input_data_path'读入待测试数据
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = BertTokenizer.from_pretrained(vocab_path)
    model_list = get_models()
    config = model_list[0].config
    G = Ensemble(config,model_list)
    df = pd.read_csv(input_data_path, encoding='utf-8', header=None)
    test_loader = create_BART_dataset(src=df[1].tolist(),
                                    tgt=df[1].tolist(),
                                    tokenizer=tokenizer,
                                    src_max_len=SRC_MAX_LEN,
                                    tgt_max_len=TGT_MAX_LEN,
                                    batch_size=BATCH_SIZE,
                                    train_mode=False,
                                    shuffle=False,
                                    pin_memory=False,
                                    num_workers=0)

    # 生成预定义数据样本
    predictions = []
    with torch.no_grad():
        for data in tqdm(test_loader):
            ids = data['source_ids'].to(device, dtype = torch.long)
            mask = data['source_mask'].to(device, dtype = torch.long)
            generated_ids = G.generate(
                input_ids = ids,
                attention_mask = mask,
                max_length=80,
                num_beams=4,
                length_penalty=0.9,
                early_stopping=True
                )
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            predictions.extend(preds)

    # 将选手程序生成的结果写入目的地地址文件中
    df[1] = predictions
    df[[0,1]].to_csv(output_data_path, header=None, index=False)

def invoke_(input_data_path, output_data_path):
    # 从输入地址 'input_data_path'读入待测试数据
    tokenizer = BertTokenizer.from_pretrained("./bart-base-ema/large_vocab.txt")
    model = BartForConditionalGeneration.from_pretrained("./bart-base-ema")
    
    df = pd.read_csv(input_data_path, encoding='utf-8', header=None)
    test_loader = create_BART_dataset(src=df[1].tolist(),
                                    tgt=df[1].tolist(),
                                    tokenizer=tokenizer,
                                    src_max_len=SRC_MAX_LEN,
                                    tgt_max_len=TGT_MAX_LEN,
                                    batch_size=BATCH_SIZE,
                                    train_mode=False,
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
            ids = data['source_ids'].to(device, dtype = torch.long)
            mask = data['source_mask'].to(device, dtype = torch.long)
            generated_ids = model.generate(
                input_ids = ids,
                attention_mask = mask,
                max_length=80,
                num_beams=4,
                length_penalty=0.9,
                early_stopping=True
                )
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            predictions.extend(preds)

    # 将选手程序生成的结果写入目的地地址文件中
    df[1] = predictions
    df[[0,1]].to_csv(output_data_path, header=None, index=False)
    
if __name__ == '__main__':
    # invoke(input_data_path='./val.csv',output_data_path='./pre2.csv')
    pass