from transformers import AutoModelForSeq2SeqLM,BertTokenizer
from transformers.optimization import get_cosine_schedule_with_warmup,AdamW
import torch
import os
import pandas as pd
import numpy as np
import random
from scripts.dataset import create_BART_dataset
from scripts.trainer import train, validate
from sklearn.model_selection import KFold

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     random.seed(seed)
     np.random.seed(seed)
     torch.backends.cudnn.deterministic = True

os.environ["WANDB_DISABLED"] = "true"
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 32
TRAIN_EPOCHS = 12
LEARNING_RATE = 1e-4
SRC_MAX_LEN = 160
TGT_MAX_LEN = 90
device = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 2222222
setup_seed(SEED)

def get_model(tokenizer):
    # tokenizer = AutoTokenizer.from_pretrained("./pretrain_model/bart-base/")
    # model = AutoModelForSeq2SeqLM.from_pretrained("./pretrain_model/bart-base/")
    model = AutoModelForSeq2SeqLM.from_pretrained("../预训练/bart-base/checkpoint-36000")

    model.resize_token_embeddings(len(tokenizer))
    # model.load_state_dict(torch.load('./saved_models/ema/best_cider_model.pkl')['model'])
    model.config.bos_token_id = tokenizer.cls_token_id
    model.config.eos_token_id = tokenizer.sep_token_id
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.forced_eos_token_id = tokenizer.sep_token_id

    return model

def main():
    kfold = 10
    tokenizer = BertTokenizer.from_pretrained("../预训练/tokenizer/")
    # 数据加载
    df1 = pd.read_csv('./data/train.csv',encoding='utf-8',header=None)
    df2 = pd.read_csv('./data/semi_train.csv',encoding='utf-8',header=None)
    df = pd.concat([df1,df2[[0,1,2]]],axis=0).reset_index(drop=True)
    print("all data: {}".format(df.shape[0]))
    kf = KFold(n_splits=kfold, random_state=SEED, shuffle=True)
    for n_fold,(tr_idx, val_idx) in enumerate(kf.split(df)):
        
        print(f"train {n_fold+1} fold model")
        train_dataset=df.iloc[tr_idx].reset_index(drop=True)
        val_dataset=df.iloc[val_idx].reset_index(drop=True)

        print("train data: {}".format(train_dataset.shape[0]))
        print("valid data: {}".format(val_dataset.shape[0]))

        train_loader = create_BART_dataset(src=train_dataset[1].tolist(),
                                        tgt=train_dataset[2].tolist(),
                                        tokenizer=tokenizer,
                                        src_max_len=SRC_MAX_LEN,
                                        tgt_max_len=TGT_MAX_LEN,
                                        batch_size=TRAIN_BATCH_SIZE,
                                        shuffle=True,
                                        pin_memory=True,
                                        num_workers=0)
        valid_loader = create_BART_dataset(src=val_dataset[1].tolist(),
                                        tgt=val_dataset[2].tolist(),
                                        tokenizer=tokenizer,
                                        src_max_len=SRC_MAX_LEN,
                                        tgt_max_len=TGT_MAX_LEN,
                                        batch_size=VALID_BATCH_SIZE,
                                        shuffle=False,
                                        pin_memory=True,
                                        num_workers=0)
        print(f"train loader:{len(train_loader)}, valid loader:{len(valid_loader)}")
        num_train_steps = int(len(train_loader) * TRAIN_EPOCHS)

        model = get_model(tokenizer)
        
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        param_optimizer = list(model.named_parameters())
        optimizer_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        ]
        optimizer = AdamW(params =  optimizer_parameters, lr=LEARNING_RATE)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(num_train_steps*0.1),
            num_training_steps=num_train_steps)

        model = torch.nn.parallel.DataParallel(model.to(device))

        from scripts.trick import EMA,FGM
        fgm = FGM(model)
        start_fgm = 5

        ema = EMA(model, 0.999)
        ema.register()
        start_ema = 5

        bst_cider = 0
        bst_bleu = 0
        bst_score = 0

        print('开始训练模型...')
        for epoch in range(0,TRAIN_EPOCHS):
            print(f'第{epoch+1}轮训练...')
            model = train(epoch, tokenizer, model, device, train_loader, optimizer,scheduler,ema,start_ema,fgm,start_fgm)
            lr = optimizer.state_dict()['param_groups'][0]['lr']
            print(f'第{epoch+1}轮验证...')
            if epoch+1>start_ema:
                ema.apply_shadow()
            cider_score, bleu_score = validate(epoch, tokenizer, model, device, valid_loader)
            print(f"Epoch:{epoch+1}, cider_score:{cider_score}, bleu_score:{bleu_score}, lr:{lr}")
            if bst_cider<cider_score:
                bst_cider = cider_score
                torch.save({'model': model.module.state_dict(),
                            'score':(2*cider_score + bleu_score)/3,
                            'bleu_score':bleu_score,
                            'cider_score':cider_score},
                        f'./saved_models/kfold/best_cider_model_{n_fold+1}.pkl')
                print(f"保存模型：第{epoch+1}轮，当前最好cider_score：{bst_cider}")
                
            if epoch+1>start_ema:
                ema.restore()
                
if __name__=='__main__':
    main()