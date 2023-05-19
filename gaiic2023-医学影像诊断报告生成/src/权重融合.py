from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,BertTokenizer
from scripts.dataset import create_BART_dataset
import pandas as pd
import os
from scripts.utils import setup_seed

os.environ["WANDB_DISABLED"] = "true"
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 16
TRAIN_EPOCHS = 50
LEARNING_RATE = 5e-5
SRC_MAX_LEN = 200
TGT_MAX_LEN = 90
SEED = 2222222
setup_seed(SEED)

# tokenizer = AutoTokenizer.from_pretrained("./pretrain_model/bart-base/")
tokenizer = BertTokenizer.from_pretrained("./tokenizer/")
model = AutoModelForSeq2SeqLM.from_pretrained("./pretrain_model/bart-base/")
model.resize_token_embeddings(len(tokenizer))
model.config.bos_token_id = tokenizer.cls_token_id
model.config.eos_token_id = tokenizer.sep_token_id
model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.forced_eos_token_id = tokenizer.sep_token_id

checkpoint_list = ['./saved_models/kfold/best_cider_model_1.pkl','./saved_models/kfold/best_cider_model_2.pkl','./saved_models/kfold/best_cider_model_3.pkl']
weight_list = [1/3,1/3,1/3]
import torch
import torch.nn as nn
def apply_swa(model: nn.Module,
              checkpoint_list: list,
              weight_list: list,
              strict: bool = True):
    checkpoint_tensor_list = [torch.load(f, map_location='cpu') for f in checkpoint_list]
    for name, param in model.named_parameters():
        try:
            param.data = sum([ckpt['model'][name] * w for ckpt, w in zip(checkpoint_tensor_list, weight_list)])
        except KeyError:
            if strict:
                raise KeyError(f"Can't match '{name}' from checkpoint")
            else:
                print(f"Can't match '{name}' from checkpoint")
    return model

swa_model = apply_swa(model,checkpoint_list=checkpoint_list,weight_list=weight_list,strict=True)
torch.save({'model': model.state_dict()}, './saved_models/kfold/swa_model.pkl')