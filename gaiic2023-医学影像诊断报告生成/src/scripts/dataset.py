from torch.utils.data.dataloader import DataLoader,Dataset
import torch
import random
import numpy as np

class Bartdataset(Dataset):
    def __init__(self,src,tgt,tokenizer,max_source_length=400, max_target_length=200,train_mode=True):
        assert len(src)==len(tgt)
        self.src = src
        self.tgt = tgt
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.tokenizer = tokenizer
        self.train_mode = train_mode
        
    def __len__(self):
        return len(self.src)
    
    # 随机替换
    def replace_token(self, tgt):
        if random.random()>0.5:
            tgt = [t if random.random()>0.3 else random.choice(tgt) for t in tgt]
            
        return tgt
    
    def __getitem__(self, index):
        src = self.src[index].split()[:self.max_source_length]
        tgt = self.tgt[index].split()[:self.max_target_length]
        
        if self.train_mode:
            tgt = self.replace_token(tgt)
        
        source_ids = [self.tokenizer.cls_token_id] + self.tokenizer.convert_tokens_to_ids(src) + [self.tokenizer.sep_token_id]
        target_ids = [self.tokenizer.cls_token_id] + self.tokenizer.convert_tokens_to_ids(tgt) + [self.tokenizer.sep_token_id]

        return {'source_ids': source_ids,'target_ids': target_ids}

from scripts.utils import data4BIO
class Bart4BIOdataset(Dataset):
    def __init__(self,src,tgt,tokenizer,max_source_length=160, max_target_length=80,train_mode=True):
        assert len(src)==len(tgt)
        self.src = src
        self.tgt = tgt
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.tokenizer = tokenizer
        self.train_mode = train_mode

    def __len__(self):
        return len(self.src)
    # 随机替换
    def replace_token(self, tgt):
        if random.random()>0.5:
            tgt = [t if random.random()>0.3 else random.choice(tgt) for t in tgt]
            
        return tgt
    def __getitem__(self, index):
        src = self.src[index].split()[:self.max_source_length]
        tgt = self.tgt[index].split()[:self.max_target_length]
        if self.train_mode:
            tgt = self.replace_token(tgt)

        source_ids = [self.tokenizer.cls_token_id] + self.tokenizer.convert_tokens_to_ids(src) + [self.tokenizer.sep_token_id]
        target_ids = [self.tokenizer.cls_token_id] + self.tokenizer.convert_tokens_to_ids(tgt) + [self.tokenizer.sep_token_id]
        src_label, tgt_label = data4BIO(source_ids,target_ids)
        src_label[0] = src_label[-1] = -100
        tgt_label[0] = tgt_label[-1] = -100
        return {'source_ids':source_ids,
                'target_ids':target_ids,
                'src_label':src_label,
                'tgt_label':tgt_label}

def paddingList(ls:list,pad_id=0,returnTensor=False):
    ls=ls[:]#不要改变了原list尺寸
    maxLen=max([len(i) for i in ls])
    for i in range(len(ls)):
        ls[i]=ls[i]+[pad_id]*(maxLen-len(ls[i]))
    return torch.LongTensor(ls) if returnTensor else ls

def collate(batch):
    source_ids = [i['source_ids'] for i in batch]
    source_ids = paddingList(source_ids,0,returnTensor=True)
    source_mask = (source_ids!=0).to(dtype=torch.long)
    target_ids = [i['target_ids'] for i in batch]
    target_ids = paddingList(target_ids,0,returnTensor=True)
    target_mask = (target_ids!=0).to(dtype=torch.long)
    return {'source_ids':source_ids,
            'source_mask':source_mask,
            'target_ids': target_ids,
            'target_mask':target_mask,
            }

def collate_bio(batch):
    src_label = [i['src_label'] for i in batch]
    tgt_label = [i['tgt_label'] for i in batch]
    src_label = paddingList(src_label,-100,returnTensor=True)
    tgt_label = paddingList(tgt_label,-100,returnTensor=True)
    source_ids = [i['source_ids'] for i in batch]
    source_ids = paddingList(source_ids,0,returnTensor=True)
    source_mask = (source_ids!=0).to(dtype=torch.long)
    target_ids = [i['target_ids'] for i in batch]
    target_ids = paddingList(target_ids,0,returnTensor=True)
    target_mask = (target_ids!=0).to(dtype=torch.long)
    return {'source_ids':source_ids,
            'source_mask':source_mask,
            'target_ids': target_ids,
            'target_mask':target_mask,
            'src_label':src_label,
            'tgt_label':tgt_label,
            }
    
def create_BART_dataset(src,
                      tgt,
                      tokenizer,
                      src_max_len=160,
                      tgt_max_len=80,
                      batch_size=1,
                      shuffle=True,
                      pin_memory=True,
                      num_workers=0,
                      train_mode=True):
    dataset = Bartdataset(src=src,
                          tgt=tgt,
                          tokenizer=tokenizer,
                          max_source_length=src_max_len,
                          max_target_length=tgt_max_len,
                          train_mode=train_mode)
    
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            collate_fn=collate,
                            pin_memory=pin_memory,
                            num_workers=num_workers)
    
    return dataloader

def create_BIO4BART_dataset(src,
                      tgt,
                      tokenizer,
                      src_max_len=160,
                      tgt_max_len=80,
                      batch_size=1,
                      shuffle=True,
                      pin_memory=True,
                      num_workers=0,
                      train_mode=True):
    dataset = Bart4BIOdataset(src=src,
                              tgt=tgt,
                              tokenizer=tokenizer,
                              max_source_length=src_max_len,
                              max_target_length=tgt_max_len,
                              train_mode=train_mode)
    
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            collate_fn=collate_bio,
                            pin_memory=pin_memory,
                            num_workers=num_workers)
    
    return dataloader