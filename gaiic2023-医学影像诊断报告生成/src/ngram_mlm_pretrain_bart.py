from transformers import AutoModelForMaskedLM,BertTokenizer
from torch.utils.data.dataloader import Dataset,DataLoader
from transformers.optimization import get_cosine_schedule_with_warmup,get_linear_schedule_with_warmup
from torch.optim import AdamW
import torch
import os
import random
import numpy as np
from tqdm import tqdm
from scripts.utils import setup_seed

os.environ["WANDB_DISABLED"] = "true"

class Config():
    batch_size = 128
    lr = 1e-4
    warmup_rate = 0.1
    train_epoch = 50
    print_step = ''
    eval_step = ''
    seed = 2222222

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    block_size = 160
    corpus_path = './data/large_corpus.txt'

    vocab_path = './tokenizer/large_vocab.txt'
    model_path = '../pretrained_model/bart-base'
    checkpoint = None
    save_model_path = './pretrain_model/DAE_seq2seq_large_last'

def build_model(config):
    tokenizer = BertTokenizer.from_pretrained(config.vocab_path)
    model = AutoModelForMaskedLM.from_pretrained(config.model_path)
    model.resize_token_embeddings(len(tokenizer))
    model.config.bos_token_id = tokenizer.cls_token_id
    model.config.eos_token_id = tokenizer.sep_token_id
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.forced_eos_token_id = tokenizer.sep_token_id
    
    return tokenizer, model

def paddingList(ls:list,val,returnTensor=False):
    ls=ls[:]
    maxLen=max([len(i) for i in ls])
    for i in range(len(ls)):
        ls[i]=ls[i]+[val]*(maxLen-len(ls[i]))
    return torch.LongTensor(ls) if returnTensor else ls

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

        self.transform_fun = {'random_mask':self.random_mask,
                              'random_deletion':self.random_deletion,
                              'random_permute':self.random_permute,
                              'random_span':self.random_span,
                              'random_other':self.random_other}

    def __len__(self):
        return len(self.examples)
    
    def random_mask(self,text_ids):
        input_ids = []
        rands = np.random.random(len(text_ids))
        idx=0
        while idx<len(rands):
            if rands[idx]<0.15:#需要mask
                ngram=np.random.choice([3,4,5], p=[0.3,0.4,0.3])#若要mask，进行x_gram mask的概率
                if ngram==5 and len(rands)<10:#太大的gram不要应用于过短文本
                    ngram=4
                if ngram==4 and len(rands)<8:
                    ngram=3
                if ngram==3 and len(rands)<6:
                    ngram=2
                if len(rands)<4:
                    ngram = 1
                L=idx+1
                R=idx+ngram#最终需要mask的右边界（开）
                while L<R and L<len(rands):
                    rands[L]=np.random.random()*0.15#强制mask
                    L+=1
                idx=R
                if idx<len(rands):
                    rands[idx]=1#禁止mask片段的下一个token被mask，防止一大片连续mask
            idx+=1
        for r, i in zip(rands, text_ids):
            if r < 0.15 * 0.8:
                input_ids.append(self.tokenizer.mask_token_id)
            elif r < 0.15 * 1:
                input_ids.append(np.random.randint(5,len(self.tokenizer)))
            else:
                input_ids.append(i)

        return input_ids
    
    def random_deletion(self, text_ids):
        new_text_ids = []
        rands = np.random.random(len(text_ids))
        for i in range(len(text_ids)):# 删除
            if rands[i]<0.15:
                continue
            new_text_ids.append(text_ids[i])

        return new_text_ids

    def random_permute(self, text_ids):
        stop_word = self.tokenizer.convert_tokens_to_ids('10')
        sentences = []
        tmp = []
        for w in text_ids:
            tmp.append(w)
            if w == stop_word:
                sentences.append(tmp)
                tmp = []
        if len(sentences)>1:
            num_sentences = len(sentences)
            num_to_permute = num_sentences
            substitutions = torch.randperm(num_sentences)[:num_to_permute]
            new_sentences = [sentences[idx] for idx in substitutions]
            text_ids = sum(new_sentences, [])
        
        return text_ids
    
    def random_span(self, text_ids):# FOR text infilling
        if len(text_ids)>10:# 多个词替换成一个[MASK]
            span_num = random.choice([0,2,3,4,5])
            idx = random.choice(range(len(text_ids)-span_num))
            text_ids = text_ids[:idx] + [self.tokenizer.mask_token_id] + text_ids[idx+span_num:]

        return text_ids
    
    def random_other(self, text_ids):

        return text_ids

    def __getitem__(self, i):
        text = self.examples[i].strip().split() #预处理，mask等操作
        text = text[:self.block_size]
        text_ids = self.tokenizer.convert_tokens_to_ids(text)
        labels = text_ids + [self.tokenizer.sep_token_id]
        # DAE
        key = np.random.choice(['random_mask','random_span','random_other'],p=[0.6,0.3,0.1])
        text_ids = self.transform_fun[key](text_ids)
        input_ids = [self.tokenizer.cls_token_id] + text_ids + [self.tokenizer.sep_token_id]

        return {'input_ids':input_ids,'labels':labels}

def collate(batch):
    pad_token_id = 0
    input_ids=[i['input_ids'] for i in batch]
    labels=[i['labels'] for i in batch]
    input_ids=paddingList(input_ids,pad_token_id,returnTensor=True)
    labels=paddingList(labels,-100,returnTensor=True)
    attention_mask=(input_ids!=pad_token_id)

    return {'input_ids':input_ids,'attention_mask':attention_mask,'labels':labels}

def save_model(model, tokenizer, save_path):
    os.makedirs(save_path, exist_ok=True)
    model_to_save = model.module if hasattr(model, 'module') else model
    torch.save(model_to_save.state_dict(), os.path.join(save_path,'pytorch_model.bin'))
    model_to_save.config.to_json_file(os.path.join(save_path,'config.json'))
    tokenizer.save_vocabulary(save_path)

def build_data(config, tokenizer):
    # 加载要预训练的文本数据
    train_dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=config.corpus_path,
        block_size=config.block_size
    )
    train_loader = DataLoader(dataset=train_dataset,
                            batch_size=config.batch_size,
                            shuffle=True,
                            collate_fn=collate,
                            num_workers=0,
                            pin_memory=True)
    return train_loader

def build_optim(config, model):
    no_decay = ["bias", "LayerNorm"]
    param_optimizer = list(model.named_parameters())
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]
    optimizer = AdamW(params =  optimizer_parameters, betas=(0.9,0.95),lr=config.lr, eps=1e-6)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(config.num_train_steps*config.warmup_rate),
        num_training_steps=config.num_train_steps)

    return optimizer, scheduler

def train(config):
    tokenizer, model = build_model(config)
    train_loader = build_data(config, tokenizer)
    num_train_steps = config.num_train_steps = int(len(train_loader) * config.train_epoch)
    print(f"train loader:{len(train_loader)}")
    print(f"tatol num steps:{num_train_steps}, num_warmup_steps:{int(num_train_steps*config.warmup_rate)}")
    device = config.device
    model = model.to(device)
    optimizer, scheduler = build_optim(config, model)
    print_step = config.print_step if config.print_step else len(train_loader) #2000
    step = 0
    batch_loss = 0
    print('开始训练模型...')
    for epoch in range(config.train_epoch):
        print(f'第{epoch+1}轮训练...')
        model.train()
        for data in tqdm(train_loader):
            step += 1
            input_ids = data['input_ids'].to(device, dtype = torch.long)
            attention_mask = data['attention_mask'].to(device, dtype = torch.long)
            labels = data['labels'].to(device, dtype = torch.long)
            loss = model(input_ids = input_ids, attention_mask = attention_mask, labels=labels).loss
            loss = loss.mean()
            batch_loss += loss.item()
            if step%print_step==0:
                lr = optimizer.state_dict()['param_groups'][0]['lr']
                print(f'Epoch: {epoch+1}, step:{step+1}, Loss:  {batch_loss/print_step}, lr:{lr}')
                batch_loss = 0
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        save_path = config.save_model_path
        save_model(model, tokenizer, save_path)

        if epoch in [25,30,35,40,46]:
            save_path = f'{config.save_model_path}-{epoch}'
            save_model(model, tokenizer, save_path)
        
def main():
    config = Config()
    setup_seed(config.seed)
    train(config)

if __name__=='__main__':
    main()