import pandas as pd

train = pd.read_csv('./data/train.csv',header=None)
test_a = pd.read_csv('./data/preliminary_a_test.csv',header=None)
test_b = pd.read_csv('./data/preliminary_b_test.csv',header=None)
semi_train = pd.read_csv('./data/semi_train.csv',header=None)
sentences = train[1].tolist() + train[2].tolist() + test_a[1].tolist() + test_b[1].tolist() + semi_train[1].tolist() + semi_train[2].tolist()

df = pd.concat([train,semi_train[[0,1,2]]],axis=0).reset_index(drop=True)
df.to_csv('./data/train_data.csv',index=False,header=None)

with open(file='./data/corpus.txt', mode='w', encoding='utf-8') as f:
    for sent in sentences:
        f.write(sent+'\n')
