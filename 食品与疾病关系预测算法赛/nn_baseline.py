import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import f1_score, fbeta_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold as KFold
from sklearn.decomposition import PCA
from torch.utils.data import Dataset, DataLoader
from torch.nn import DataParallel

# 降维
def 降维(feats, nfeats=64):
    pca = PCA(n_components=nfeats,random_state=2023)
    new_feats = pca.fit_transform(feats)
    
    return new_feats

disease_feature1 = pd.read_csv('/kaggle/input/food-disease/train/训练集/disease_feature1.csv')
disease_feature2 = pd.read_csv('/kaggle/input/food-disease/train/训练集/disease_feature2.csv')
disease_feature3 = pd.read_csv('/kaggle/input/food-disease/train/训练集/disease_feature3.csv')
new_feat1 = 降维(disease_feature1.iloc[:,1:], nfeats=128)
new_feat2 = 降维(disease_feature2.iloc[:,1:], nfeats=128)
new_feat3 = 降维(disease_feature3.iloc[:,1:], nfeats=128)

feat1 = pd.DataFrame(new_feat1)
feat1['disease_id'] = disease_feature1.disease_id

feat2 = pd.DataFrame(new_feat2)
feat2['disease_id'] = disease_feature2.disease_id

feat3 = pd.DataFrame(new_feat3)
feat3['disease_id'] = disease_feature3.disease_id

# 数据读取
test_food = pd.read_csv('/kaggle/input/food-disease/test_A/初赛A榜测试集/preliminary_a_food.csv')
test_sub = pd.read_csv('/kaggle/input/food-disease/test_A/初赛A榜测试集/preliminary_a_submit_sample.csv')
train_food = pd.read_csv('/kaggle/input/food-disease/train/训练集/train_food.csv')
train_answer = pd.read_csv('/kaggle/input/food-disease/train/训练集/train_answer.csv')

train = train_answer.merge(train_food, on='food_id', how='left').merge(feat1, on='disease_id', how='left').merge(feat2, on='disease_id', how='left').merge(feat3, on='disease_id', how='left')
test = test_sub.merge(test_food, on='food_id', how='left').merge(feat1, on='disease_id', how='left').merge(feat2, on='disease_id', how='left').merge(feat3, on='disease_id', how='left')

# 突然想删一下
del disease_feature1
del disease_feature2
del disease_feature3

del new_feat1
del new_feat2
del new_feat3

del feat1
del feat2
del feat3

del test_food
del train_food
del train_answer

# 减少一下特征 N_x
select_feats = ['food_id','disease_id','related']
for col, value in zip(train.isna().describe().iloc[2].index[3:],train.isna().describe().iloc[2].values[3:]):
    if value==True:
        continue
    select_feats.append(col)
    
# len(list(filter(lambda x:'N' in str(x),select_feats))) # 69
train = train[select_feats]
select_feats[2] = 'related_prob'
test = test[select_feats]

# 填上自己的幸运值
train = train.fillna(0)
test = test.fillna(0)

# 统计一下 disease_id
disease_ids = list(set(test.disease_id.tolist() + train.disease_id.tolist()))
print(f'all disease type num is {len(disease_ids)}')
disease2idx = {disease:i for i, disease in enumerate(disease_ids)}

# nn model
import torch
import torch.nn as nn
import torch.nn.functional as F

class NN_Baseline(nn.Module):
    def __init__(self):
        super(NN_Baseline, self).__init__()
        # 处理 disease_id
        self.emb = nn.Embedding(407, 64)
        # 处理 N_x
        self.fc_Nx = nn.Sequential(nn.Linear(69, 64),nn.ReLU())
        # 处理 F_x
        self.fc_F1 = nn.Sequential(nn.Linear(128, 128),nn.ReLU())
        self.fc_F2 = nn.Sequential(nn.Linear(128, 128),nn.ReLU())
        self.fc_F3 = nn.Sequential(nn.Linear(128, 128),nn.ReLU())
        
        self.fusion = nn.Sequential(nn.Linear(128*5, 256),
                                    nn.ReLU(),
                                    nn.Linear(256, 256),
                                    nn.ReLU())
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Sequential(nn.Linear(256, 1),nn.Sigmoid())
    
    def forward(self, x1,x2,x3,x4,x5):
        out1 = torch.mean(self.emb(x1),dim=1)
        out2 = self.fc_Nx(x2)
        out3 = self.fc_F1(x3)
        out4 = self.fc_F2(x4)
        out5 = self.fc_F3(x5)
        
        # 随便处理一下，没有技巧，全靠感觉
        out_34 = out3 * out4
        out_35 = out3 * out5
        out_45 = out4 * out5
        out = torch.cat([out1,out2,out3+out4+out5,out_34,out_35,out_45], dim=-1)
        
        out = self.fusion(out)
        out = self.fc(self.dropout(out))
        return out
    
class NNDataset(Dataset):
    def __init__(self, df, disease2idx, train_mode=True):
        super(NNDataset, self).__init__()
        self.train_mode = train_mode
        self.df = df
        self.disease2idx = disease2idx
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        sample = self.df.iloc[idx].values
        x1 = np.array([self.disease2idx[sample[1]]])
        x2 = sample[3:3+69].astype(np.float32)
        x3 = sample[3+69:3+69+128].astype(np.float32)
        x4 = sample[3+69+128:3+69+128*2].astype(np.float32)
        x5 = sample[3+69+128*2:].astype(np.float32)
        label = np.array([sample[2]],dtype=np.float32)[0]
        
        return x1,x2,x3,x4,x5,label
    
# 评估函数
def evaluate_accuracy(x1,x2,x3,x4,x5, y,net,criterion):
    net.eval()
    with torch.no_grad():
        out = net(x1,x2,x3,x4,x5)
        out = out.squeeze(dim=-1)
        loss = criterion(out,y).item()
    out = out.squeeze(dim=-1)
    correct= (out.ge(0.5) == y).sum().item()
    n = y.shape[0]

    return correct / n, loss

# 五折一下
folds = KFold(n_splits=5, shuffle=True, random_state=2023)
predictions = np.zeros([len(test),])

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train, train.related)):
    print("fold n°{}".format(fold_+1))
    trn_data = train.iloc[trn_idx].reset_index(drop=True)
    val_data = train.iloc[val_idx].reset_index(drop=True)
    
    train_set = NNDataset(trn_data, disease2idx)
    train_loader = DataLoader(train_set, batch_size=256, shuffle=True)
    
    valid_set = NNDataset(val_data, disease2idx,train_mode=False)
    valid_loader = DataLoader(valid_set, batch_size=256, shuffle=False)
    
    criterion = nn.BCELoss().cuda()
    model = NN_Baseline()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model = DataParallel(model.cuda())
    print(f'总迭代步数{len(train_loader)}')
    best_acc = 0
    for e in range(1,16):
        for i, (x1,x2,x3,x4,x5, y) in enumerate(train_loader):
            model.train()
            x1,x2,x3,x4,x5, y = x1.cuda(),x2.cuda(),x3.cuda(),x4.cuda(),x5.cuda(), y.cuda(0)
            out = model(x1,x2,x3,x4,x5)
            out = out.squeeze(dim=-1)
            loss = criterion(out, y)
            if i%100==0 or i==0:
                train_acc, _ = evaluate_accuracy(x1,x2,x3,x4,x5, y, model, criterion)
                print(f'epoch:{e},step:{i},loss:{loss.item()},train acc:{train_acc}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 每轮训练结束后验证一次
        acc_avg = 0
        loss_avg = 0
        n = len(valid_loader)
        for (x1,x2,x3,x4,x5, y) in valid_loader:
            valid_acc,valid_loss = evaluate_accuracy(x1.cuda(),x2.cuda(),x3.cuda(),x4.cuda(),x5.cuda(), y.cuda(), model, criterion)
            acc_avg += valid_acc/n
            loss_avg += valid_loss/n
        print(f'epoch:{e},valid loss:{loss_avg},valid acc:{acc_avg}')
        if best_acc<acc_avg:
            best_acc = acc_avg
            print(f'saved model epoch {e}')
            torch.save(model.state_dict(),f'/kaggle/working/best_fold{fold_+1}.model')

# 加载模型
model_list = []
for n in range(1, 5+1):
    model = NN_Baseline().cuda()
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(f'/kaggle/working/best_fold{n}.model'))
    model.eval()
    model_list.append(model)

# 突然想起验证一下，也不知道写得对不对
valid_set = NNDataset(train, disease2idx,train_mode=False)
valid_loader = DataLoader(valid_set, batch_size=512, shuffle=False)
y_valid = train.related
pred_prob = []
with torch.no_grad():
    for (x1,x2,x3,x4,x5, y) in valid_loader:
        out = 0
        for model in model_list:
            out += model(x1.cuda(),x2.cuda(),x3.cuda(),x4.cuda(),x5.cuda())/folds.n_splits
        pred_prob += out.squeeze(dim=-1).tolist()

y_pre = [int(x>0.2) for x in pred_prob]
acc = [int(i==j) for i,j in zip(y_valid, y_pre)]
acc = sum(acc)/len(acc)
f1 = f1_score(y_valid, y_pre,average='binary')
precision = precision_score(y_valid, y_pre,average='binary')
recall = recall_score(y_valid, y_pre,average='binary')
fpr,tpr,threshold = roc_curve(y_valid, y_pre)
roc_auc = auc(fpr,tpr)
print(f"roc_auc:{roc_auc}")
print(f"acc:{acc}")
print(f"F1 score: {f1}")
print(f"Precision score: {precision}")
print(f"Recall score: {recall}")
print(f"(F1+auc)/2: {(f1+roc_auc)/2}")

# 测试推理
valid_set = NNDataset(test, disease2idx,train_mode=False)
valid_loader = DataLoader(valid_set, batch_size=512, shuffle=False)
pred_prob = []
with torch.no_grad():
    for (x1,x2,x3,x4,x5, y) in valid_loader:
        out = 0
        for model in model_list:
            out += model(x1.cuda(),x2.cuda(),x3.cuda(),x4.cuda(),x5.cuda())/folds.n_splits
        pred_prob += out.squeeze(dim=-1).tolist()

# 没有什么机智的后处理
test_sub['related_prob'] = [x+0.2 for x in pred_prob]
test_sub.to_csv('/kaggle/working/nn_baseline.csv', index=False)
