'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2024-01-07 15:12:19
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2024-01-12 15:47:29
FilePath: /codes/LLMRec/DALR/encoder/models/loss_utils.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch as t
import torch.nn.functional as F
import torch
import sys



def cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds):
    pos_preds = (anc_embeds * pos_embeds).sum(-1)
    neg_preds = (anc_embeds * neg_embeds).sum(-1)
    return t.sum(F.softplus(neg_preds - pos_preds))


def reg_pick_embeds(embeds_list):
    reg_loss = 0
    for embeds in embeds_list:
        reg_loss += embeds.square().sum()
    return reg_loss


def cal_infonce_loss(embeds1, embeds2, all_embeds2, temp=1.0):
    normed_embeds1 = embeds1 / t.sqrt(1e-8 + embeds1.square().sum(-1, keepdim=True))
    normed_embeds2 = embeds2 / t.sqrt(1e-8 + embeds2.square().sum(-1, keepdim=True))
    
    normed_all_embeds2 = all_embeds2 / t.sqrt(1e-8 + all_embeds2.square().sum(-1, keepdim=True))
    nume_term = -(normed_embeds1 * normed_embeds2 / temp).sum(-1)
    deno_term = t.log(t.sum(t.exp(normed_embeds1 @ normed_all_embeds2.T / temp), dim=-1))
    cl_loss = (nume_term + deno_term).sum()
    return cl_loss


def cal_mim_loss(anchor, positive, negatives, temperature = 1.0):
        # anchor: 嵌入向量的锚点
        # positive: 正样本的嵌入向量
        # negatives: 负样本的嵌入向量集合
        # 计算正样本和锚点之间的相似度
        pos_similarity = F.cosine_similarity(anchor, positive, dim=-1) / temperature

        # 计算负样本和锚点之间的相似度
        neg_similarity = F.cosine_similarity(anchor, negatives, dim=-1) / temperature
        # 计算互信息最大化损失
        loss = -torch.log(torch.exp(pos_similarity) / (torch.exp(pos_similarity) + torch.exp(neg_similarity)) )
        return loss


def reg_params(model):
    reg_loss = 0
    for W in model.parameters():
        reg_loss += W.norm(2).square()
    return reg_loss


def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
    loss = loss.mean()
    return loss


def ssl_con_loss(x, y, temp=1.0):
    x = F.normalize(x)
    y = F.normalize(y)
    mole = t.exp(t.sum(x * y, dim=1) / temp)
    deno = t.sum(t.exp(x @ y.T / temp), dim=1)
    return -t.log(mole / (deno + 1e-8) + 1e-8).mean()


def alignment(x, y, alpha=2):
    x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()


def uniformity(x):
    x = F.normalize(x, dim=-1)
    return t.pdist(x, p=2).pow(2).mul(-2).exp().mean().log()