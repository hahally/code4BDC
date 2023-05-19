from transformers import BertTokenizer
from transformers.models.bart.modeling_bart import BartForConditionalGeneration
from transformers.optimization import get_linear_schedule_with_warmup,get_cosine_schedule_with_warmup
import torch
from torch.optim import AdamW
import os
import pandas as pd
from scripts.utils import setup_seed,Sparsemax
from scripts.dataset import create_BART_dataset
from scripts.evaluate import CiderD
from tqdm import tqdm
from scripts.trick import EMA,FGM
from nltk.translate.bleu_score import corpus_bleu

os.environ["WANDB_DISABLED"] = "true"
class Config():
    train_batch_size = 32
    valid_batch_size = 32
    lr = 5e-5
    warmup_rate = 0.1
    train_epoch = 20
    print_step = 500
    eval_step = ''
    seed = 2222222
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    src_max_len = 160
    tgt_max_len = 90
    
    train_data_path = './data/train_data.csv'
    train_size = 1-0.075
    
    vocab_path = '../预训练/bart-base/tokenizer/vocab.txt'
    model_path = "../预训练/bart-base/checkpoint-36000/"
    checkpoint = None
    save_model_path = './saved_models/ngram_add_label'
    
    start_ema = 1000
    start_fgm = 1000

def build_model(config):
    tokenizer = BertTokenizer.from_pretrained(config.vocab_path)
    # model = BartForConditionalGeneration.from_pretrained(config.model_path)
    from scripts.models import BartForConditionalGenerationWithCopyMech
    model = BartForConditionalGenerationWithCopyMech.from_pretrained(config.model_path)
    model.resize_token_embeddings(len(tokenizer))
    if config.checkpoint:
        model.load_state_dict(torch.load(config.checkpoint)['model'],strict=False)

    model.config.bos_token_id = tokenizer.cls_token_id
    model.config.eos_token_id = tokenizer.sep_token_id
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.forced_eos_token_id = tokenizer.sep_token_id
    
    return tokenizer, model

def build_data(config):
    # 数据加载
    df = pd.read_csv(config.train_data_path,encoding='utf-8',header=None)
    train_size = config.train_size
    SEED = config.seed
    train_dataset=df.sample(frac=train_size,random_state = SEED)
    val_dataset=df.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)

    print("all data: {}".format(df.shape[0]))
    print("train data: {}".format(train_dataset.shape[0]))
    print("valid data: {}".format(val_dataset.shape[0]))
    
    return train_dataset, val_dataset

def build_optim(config, model):
    no_decay = ["bias", "LayerNorm", "LayerNorm"]
    other = ['copy_module']
    no_main = no_decay + other
    params = list(model.named_parameters())
    optimizer_parameters = [
        {'params':[p for n,p in params if not any(nd in n for nd in no_main)],'weight_decay':1e-2,'lr':2e-5},
        {'params':[p for n,p in params if not any(nd in n for nd in other) and any(nd in n for nd in no_decay) ],'weight_decay':0,'lr':2e-5},
        {'params':[p for n,p in params if any(nd in n for nd in other) and any(nd in n for nd in no_decay) ],'weight_decay':0,'lr':1e-4},
        {'params':[p for n,p in params if any(nd in n for nd in other) and not any(nd in n for nd in no_decay) ],'weight_decay':1e-2,'lr':1e-4},
    ]
    optimizer = AdamW(params =  optimizer_parameters, lr=config.lr, eps=1e-7)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(config.num_train_steps*config.warmup_rate),
        num_training_steps=config.num_train_steps)
    
    return optimizer, scheduler

def save_model(model, save_path):
    model_to_save = model.module if hasattr(model, 'module') else model
    torch.save(model_to_save.state_dict(), os.path.join(save_path,'pytorch_model.bin'))
    model_to_save.config.to_json_file(os.path.join(save_path,'config.json'))

def validate(tokenizer, model, device, loader,gts=None,ref=None):
    model.eval()
    res = []
    hyp = []
    tot = 0
    with torch.no_grad():
        for data in tqdm(loader):
            y = data['target_ids'].to(device, dtype = torch.long)
            ids = data['source_ids'].to(device, dtype = torch.long)
            mask = data['source_mask'].to(device, dtype = torch.long)
            generated_ids = model.generate(
                input_ids = ids,
                attention_mask = mask, 
                max_length=80, 
                num_beams=3,
                length_penalty=0.9,
                early_stopping=True,
                )
            for g,t in zip(generated_ids, y):
                pred = tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                # tgt = tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                res.append({'image_id':tot, 'caption': [pred]})
                # ref.append([tgt.split()])
                # gts[tot] = [tgt]
                hyp.append(pred.split())
                tot += 1
    bleu = corpus_bleu(list_of_references=ref,hypotheses=hyp)
    CiderD_scorer = CiderD(df='corpus', sigma=15)
    cider_score, cider_scores = CiderD_scorer.compute_score(gts, res)
    
    return cider_score, bleu

def train_epoch(config):
    tokenizer, model = build_model(config)
    train_dataset, val_dataset = build_data(config)
    ref = val_dataset[2].apply(lambda x:[x.split()]).tolist()
    gts = {tot:[tgt] for tot,tgt in enumerate(val_dataset[2].tolist())}
    
    train_loader = create_BART_dataset(src=train_dataset[1].tolist(),
                                    tgt=train_dataset[2].tolist(),
                                    tokenizer=tokenizer,
                                    src_max_len=config.src_max_len,
                                    tgt_max_len=config.tgt_max_len,
                                    batch_size=config.train_batch_size,
                                    shuffle=True,
                                    pin_memory=True,
                                    num_workers=0)
    valid_loader = create_BART_dataset(src=val_dataset[1].tolist(),
                                    tgt=val_dataset[2].tolist(),
                                    tokenizer=tokenizer,
                                    src_max_len=config.src_max_len,
                                    tgt_max_len=config.tgt_max_len,
                                    batch_size=config.valid_batch_size,
                                    train_mode=False,
                                    shuffle=False,
                                    pin_memory=True,
                                    num_workers=0)
    num_train_steps = config.num_train_steps = int(len(train_loader) * config.train_epoch)
    print(f"train loader:{len(train_loader)}, valid loader:{len(valid_loader)}")
    print(f"tatol num steps:{num_train_steps}, num_warmup_steps:{int(num_train_steps*config.warmup_rate)}")
    device = config.device
    model = model.to(device)
    
    optimizer, scheduler = build_optim(config, model)
    
    ema = EMA(model, 0.95)
    ema.register()
    start_ema = config.start_ema
    
    fgm = FGM(model,epsilon=0.125, emb_name='shared')
    start_fgm = config.start_fgm

    bst_score = 0

    eval_step = config.eval_step if config.eval_step else len(train_loader) #2000
    step = 0
    batch_loss = 0
    loss_fct = Sparsemax(k_sparse=100)
    for epoch in range(config.train_epoch):
        for data in tqdm(train_loader):
            step += 1
            model.train()
            optimizer.zero_grad()
            y = data['target_ids'].to(device, dtype = torch.long)
            y_ids = y[:, :-1].contiguous()
            lm_labels = y[:, 1:].clone().detach()
            lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
            ids = data['source_ids'].to(device, dtype = torch.long)
            mask = data['source_mask'].to(device, dtype = torch.long)
            out = model(input_ids = ids, attention_mask = mask, decoder_input_ids=y_ids, labels=lm_labels)
            logits = out.logits
            loss = loss_fct(logits.view(-1, len(tokenizer)), lm_labels.view(-1)).mean()
            loss = out.loss.mean()
            batch_loss += loss.item()
            
            loss.backward()
            if (epoch+1)>start_fgm:
                fgm.attack()
                loss_adv = model(input_ids = ids, attention_mask = mask, decoder_input_ids=y_ids, labels=lm_labels)
                logits = out.logits
                loss_adv = loss_fct(logits.view(-1, len(tokenizer)), lm_labels.view(-1)).mean()
                loss_adv.backward()
                fgm.restore()
                
            optimizer.step()
            scheduler.step()
            if epoch+1>start_ema:
                ema.update()
                
            if step%config.print_step==0:
                print(f'Epoch: {epoch+1}, step:{step}, Loss:  {batch_loss/config.print_step}')
                batch_loss = 0
                
            if step % eval_step==0 and step !=0:
                model.eval()
                lr = optimizer.state_dict()['param_groups'][0]['lr']
                if epoch+1>start_ema:
                    ema.apply_shadow()
                cider_score, bleu_score = validate(tokenizer, model, device, valid_loader, gts, ref)
                score = (2*cider_score + bleu_score)/3
                print(f"Epoch:{epoch+1}, score:{score}, cider_score:{cider_score}, bleu_score:{bleu_score}, lr:{lr}")
                
                if bst_score<score:
                    bst_score = score
                    save_model(model, config.save_model_path)
                    print(f"保存模型: step:{step}, 当前最好bst_score:{bst_score}")
                    
                if epoch+1>start_ema:
                    ema.restore()

def main():   
    config = Config()
    setup_seed(config.seed)
    train_epoch(config)

if __name__=='__main__':
    main()