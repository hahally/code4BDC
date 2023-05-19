from collections import Counter
import pandas as pd

train = pd.read_csv('./data/train.csv',header=None)
test_a = pd.read_csv('./data/preliminary_a_test.csv',header=None)
test_b = pd.read_csv('./data/preliminary_b_test.csv',header=None)
semi_train = pd.read_csv('./data/semi_train.csv',header=None)
df = pd.concat([train,semi_train[[0,1,2]]],axis=0).reset_index(drop=True)

# 生成 ngram dict
from tqdm import tqdm
from nltk import ngrams
ngram_corpus = []
token2count=Counter()
for p in range(4,5):
    for tokens in tqdm(df[1].tolist()):
        tokens = tokens.split()
        token2count.update(list(ngrams(tokens,p)))
 
ngram_count = sorted(token2count.items(), key=lambda x:x[1],reverse=True)
vocab=[]
ct = 50
for k,v in ngram_count:
    if v>=ct:
        vocab.append(k)
# vocab.sort(reverse=False)
print(len(vocab))
with open('./tokenizer/ngram_4_vocab.txt', "w", encoding="utf-8") as f:
    for i in vocab:
        f.write(' '.join(i)+'\n')