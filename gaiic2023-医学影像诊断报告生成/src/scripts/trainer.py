from tqdm import tqdm
import torch
from scripts.evaluate import CiderD

def train(epoch, tokenizer, model, device, train_loader, optimizer,scheduler,ema,start_ema,fgm,start_fgm):
    model.train()
    for _,data in enumerate(tqdm(train_loader)):
        y = data['target_ids'].to(device, dtype = torch.long)
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = data['source_ids'].to(device, dtype = torch.long)
        mask = data['source_mask'].to(device, dtype = torch.long)

        loss = model(input_ids = ids, attention_mask = mask, decoder_input_ids=y_ids, labels=lm_labels).loss
        loss = loss.mean()
        if _%500==0:
            print(f'Epoch: {epoch+1}, step:{_+1}, Loss:  {loss.item()}')
        
        optimizer.zero_grad()
        
        loss.backward()
        
        if (epoch+1)>start_fgm:
            fgm.attack()
            loss_adv = model(input_ids = ids, attention_mask = mask, decoder_input_ids=y_ids, labels=lm_labels).loss
            loss_adv = loss_adv.mean()
            loss_adv.backward()
            fgm.restore()
        
        optimizer.step()
        scheduler.step()
        if epoch+1>start_ema:
            ema.update()
    
    return model

from nltk.translate.bleu_score import corpus_bleu
def validate(epoch, tokenizer, model, device, loader):
    model.eval()
    res, gts = [], {}
    ref = []
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
                # temperature=0.9,
                length_penalty=0.8,
                # top_k=100,
                # top_p=0.99,
                early_stopping=True,
                output_hidden_states=True
                )
            for g,t in zip(generated_ids, y):
                pred = tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                tgt = tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                res.append({'image_id':tot, 'caption': [pred]})
                gts[tot] = [tgt]
                ref.append([tgt.split()])
                hyp.append(pred.split())
                tot += 1
    bleu = corpus_bleu(list_of_references=ref,hypotheses=hyp)
    CiderD_scorer = CiderD(df='corpus', sigma=15)
    cider_score, cider_scores = CiderD_scorer.compute_score(gts, res)
    
    return cider_score, bleu