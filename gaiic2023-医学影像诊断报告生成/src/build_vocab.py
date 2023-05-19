from collections import Counter
import pandas as pd

train = pd.read_csv('./data/train.csv',header=None)
test = pd.read_csv('./data/preliminary_a_test.csv',header=None)
test_b = pd.read_csv('./data/preliminary_b_test.csv',header=None)
semi_train = pd.read_csv('./data/semi_train.csv',header=None)

allData=  train[1].apply(lambda x: x.split()).tolist() + train[2].apply(lambda x: x.split()).tolist() + semi_train[1].apply(lambda x: x.split()).tolist() + semi_train[2].apply(lambda x: x.split()).tolist()
test_data = test[1].apply(lambda x: x.split()).tolist() + test_b[1].apply(lambda x: x.split()).tolist()

# 临床信息
add_info =  []
for i in semi_train[3].dropna().tolist():
    if i.strip():
        add_info.append(i.split())
# allData += add_info
ct=0
token2count=Counter()
for i in allData+test_data:
    token2count.update([int(j) for j in i])

pre=['[PAD]','[UNK]','[CLS]','[SEP]','[MASK]',]
tail=[]
for k,v in token2count.items():
    if v>=ct:
        tail.append(k)
tail.sort()
vocab=pre+tail
print(f"词频：{ct}，词表大小：{len(vocab)}")
with open('./tokenizer/vocab.txt', "w", encoding="utf-8") as f:
    for i in vocab:
        f.write(str(i)+'\n')
