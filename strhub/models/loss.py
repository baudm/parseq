import torch as T
import torch.nn.functional as F

def cross_entropy(logits, tgt, device, ignore_index=None):
    logits = logits.flatten(end_dim=1)
    tgt = tgt.flatten()
    tgt_one_hot = F.one_hot(tgt.type(T.LongTensor), logits.shape[-1]).to(device)
    loss_sum = T.sum(-tgt_one_hot * F.log_softmax(logits, -1), -1)
    if ignore_index is not None:
        ind = tgt.ne(ignore_index).nonzero(as_tuple=True)[0]
        loss_sum = loss_sum[ind]
    loss = T.mean(loss_sum)
    return loss