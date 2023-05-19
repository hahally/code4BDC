from itertools import chain
import torch
import numpy as np
import random
from torch.utils.data import Dataset,DataLoader
from transformers import BertTokenizer

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     random.seed(seed)
     np.random.seed(seed)
     torch.backends.cudnn.deterministic = True

import torch.nn as nn
# Sparse Softmax 
class Sparsemax(nn.Module):
    """Sparsemax loss"""

    def __init__(self, k_sparse=1):
        super(Sparsemax, self).__init__()
        self.k_sparse = k_sparse
        
    def forward(self, preds, labels):
        """
        Args:
            preds (torch.Tensor):  [batch_size, number_of_logits]
            labels (torch.Tensor): [batch_size] index, not ont-hot
        Returns:
            torch.Tensor
        """
        preds = preds.reshape(preds.size(0), -1) # [batch_size, -1]
        topk = preds.topk(self.k_sparse, dim=1)[0] # [batch_size, k_sparse]
        
        # log(sum(exp(topk)))
        pos_loss = torch.logsumexp(topk, dim=1)
        # s_t
        mask = labels!=-100
        labels[~mask] = 0
        neg_loss = torch.gather(preds, 1, labels[:, None].expand(-1, preds.size(1)))[:, 0]
        loss = (pos_loss - neg_loss) * mask
        return loss.sum()/mask.sum()

# 最长公共子序列
from collections import defaultdict
def longest_common_subsequence(source, target):
    """最长公共子序列（source和target的最长非连续子序列）
    返回：子序列长度, 映射关系（映射对组成的list）
    注意：最长公共子序列可能不止一个，所返回的映射只代表其中一个。
    """
    c = defaultdict(int)
    for i, si in enumerate(source, 1):
        for j, tj in enumerate(target, 1):
            if si == tj:
                c[i, j] = c[i - 1, j - 1] + 1
            elif c[i, j - 1] > c[i - 1, j]:
                c[i, j] = c[i, j - 1]
            else:
                c[i, j] = c[i - 1, j]
    l, mapping = c[len(source), len(target)], []
    i, j = len(source) - 1, len(target) - 1
    while len(mapping) < l:
        if source[i] == target[j]:
            mapping.append((i, j))
            i, j = i - 1, j - 1
        elif c[i + 1, j] > c[i, j + 1]:
            j = j - 1
        else:
            i = i - 1
    return l, mapping[::-1]

#### BIO copy 数据
import numpy as np
def data4BIO(src,tgt):
    # 0-》O 不进行复制
    # 1-》B 当前词来自原文
    # 2-》I 与前一个词组成ngram
    mapping = longest_common_subsequence(src, tgt)[1]
    src_label = [0]*len(src)
    tgt_label = [0]*len(tgt)
    i0, j0 = -2, -2
    for i, j in mapping:
        if i == i0 + 1 and j == j0 + 1:
            src_label[i] = 2
            tgt_label[j] = 2
        else:
            src_label[i] = 1
            tgt_label[j] = 1
        i0, j0 = i, j
               
    return src_label,tgt_label