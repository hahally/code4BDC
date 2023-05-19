import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_kl_loss(p, q, pad_mask=None):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
    
    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss = p_loss.masked_fill_(pad_mask, 0.)
        q_loss = q_loss.masked_fill_(pad_mask, 0.)

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.sum()/(pad_mask==0).sum()
    q_loss = q_loss.sum()/(pad_mask==0).sum()

    loss = (p_loss + q_loss) / 2
    return loss

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

class EMA():
    def __init__(self, model, decay=0.9):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

    
class FGM():
    def __init__(self, model, epsilon=0.125, emb_name='shared'):
        self.model = model
        self.epsilon = epsilon
        self.emb_name = emb_name
        self.backup = {}
    def attack(self):
        # emb_name这个参数要换成你模型中embedding的参数名
        emb_name = self.emb_name
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)
    def restore(self):
        # emb_name这个参数要换成你模型中embedding的参数名
        emb_name = self.emb_name
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

class LabelSmoothing(nn.Module):

    def __init__(self, smoothing=0., ignore_index=-100):
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction='none')
        self.smoothing = smoothing
        self.ignore_index = ignore_index

    def forward(self, inputs, targets, norm=1):
        inputs = torch.log(inputs)
        vocab_size = inputs.size(-1)
        batch_size = targets.size(0)
        length = targets.size(1)
        if self.ignore_index is not None:
            mask = (targets == self.ignore_index).view(-1)
        
        index = targets.unsqueeze(-1)
        targets = F.one_hot(targets, num_classes=inputs.size(-1))
        targets = targets * (1 - self.smoothing) + self.smoothing / vocab_size
        loss = self.criterion(inputs.view(-1, vocab_size), 
                              targets.view(-1, vocab_size).detach()).sum(dim=-1)
        if self.ignore_index is not None:
            return loss.masked_fill(mask, 0.).sum() / norm
        else:
            return loss.sum() / norm
  
class FocalLoss(torch.nn.Module):
    """Multi-class Focal loss implementation"""
    def __init__(self, gamma=2, weight=None, reduction='mean', ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        log_pt = torch.log_softmax(input, dim=1)
        pt = torch.exp(log_pt)
        log_pt = (1 - pt) ** self.gamma * log_pt
        loss = torch.nn.functional.nll_loss(log_pt, target, self.weight, reduction=self.reduction, ignore_index=self.ignore_index)
        return loss

class AWP:
    def __init__(self, model, optimizer, *, adv_param='weight',
                 adv_lr=0.001, adv_eps=0.001):
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.backup = {}
        
    def perturb(self):
        """
        Perturb model parameters for AWP gradient
        Call before loss and loss.backward()
        """
        self._save()  # save model parameters
        self._attack_step()  # perturb weights

    def _attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                grad = self.optimizer.state[param]['exp_avg']
                norm_grad = torch.norm(grad)
                norm_data = torch.norm(param.detach())

                if norm_grad != 0 and not torch.isnan(norm_grad):
                    # Set lower and upper limit in change
                    limit_eps = self.adv_eps * param.detach().abs()
                    param_min = param.data - limit_eps
                    param_max = param.data + limit_eps

                    # Perturb along gradient
                    # w += (adv_lr * |w| / |grad|) * grad
                    param.data.add_(grad, alpha=(self.adv_lr * (norm_data + e) / (norm_grad + e)))

                    # Apply the limit to the change
                    param.data.clamp_(param_min, param_max)
                    
    def _save(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.clone().detach()
                else:
                    self.backup[name].copy_(param.data)

    def restore(self):
        """
        Restore model parameter to correct position; AWP do not perturbe weights, it perturb gradients
        Call after loss.backward(), before optimizer.step()
        """
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])