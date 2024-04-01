'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2024-01-02 21:52:32
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2024-01-10 20:49:55
FilePath: /codes/LLMRec/RLMRec/encoder/models/general_cf/lightgcn.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import sys
import torch as t
from torch import nn
import torch.nn.functional as F
from config.configurator import configs
from models.loss_utils import cal_bpr_loss, reg_params
from models.base_model import BaseModel
from models.model_utils import SpAdjEdgeDrop
from sklearn.manifold import TSNE 
import pandas as pd


init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class LightGCN(BaseModel):
    def __init__(self, data_handler):
        super(LightGCN, self).__init__(data_handler)
        self.adj = data_handler.torch_adj
        self.keep_rate = configs['model']['keep_rate']
        self.user_embeds = nn.Parameter(init(t.empty(self.user_num, self.embedding_size)))
        self.item_embeds = nn.Parameter(init(t.empty(self.item_num, self.embedding_size)))
        self.edge_dropper = SpAdjEdgeDrop()
        self.final_embeds = None
        self.is_training = False

        # hyper-parameter
        self.layer_num = self.hyper_config['layer_num']
        self.reg_weight = self.hyper_config['reg_weight']
    
    
    def _propagate(self, adj, embeds):
        return t.spmm(adj, embeds)
    
    
    def forward(self, adj=None, keep_rate=1.0):
        if adj is None:
            adj = self.adj
        if not self.is_training and self.final_embeds is not None:
            return self.final_embeds[:self.user_num], self.final_embeds[self.user_num:]
        
        embeds = t.concat([self.user_embeds, self.item_embeds], axis=0)
        embeds_list = [embeds]

        if self.is_training:
            adj = self.edge_dropper(adj, keep_rate)

        """核心Layer  聚合层"""
        for i in range(self.layer_num):
            embeds = self._propagate(adj, embeds_list[-1])
            embeds_list.append(embeds)

        embeds = sum(embeds_list)
        self.final_embeds = embeds

        return embeds[:self.user_num], embeds[self.user_num:]
    
    
    def cal_loss(self, batch_data):
        self.is_training = True
        user_embeds, item_embeds = self.forward(self.adj, self.keep_rate)

        ancs, poss, negs = batch_data   ## 4096 tensor [cuda]
        anc_embeds = user_embeds[ancs]  # torch.Size([4096, 32])
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]
        
        bpr_loss = cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds) / anc_embeds.shape[0]
        reg_loss = self.reg_weight * reg_params(self)
        loss = bpr_loss + reg_loss
        losses = {'bpr_loss': bpr_loss, 'reg_loss': reg_loss}
        
        return loss, losses
    

    def full_predict(self, batch_data):
        user_embeds, item_embeds = self.forward(self.adj, 1.0)
        self.is_training = False
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()
        pck_user_embeds = user_embeds[pck_users]
        full_preds = pck_user_embeds @ item_embeds.T
        full_preds = self._mask_predict(full_preds, train_mask)
        return full_preds


    def output_entity_embedding(self, batch_data):
        user_embeds, item_embeds = self.forward(self.adj, False)
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()
        pck_user_embeds = user_embeds[pck_users]

        # tensor cuda To numpy
        pck_users = pck_users.cpu().detach().numpy()
        pck_user_embeds = pck_user_embeds.cpu().detach().numpy()

        df_emb = pd.DataFrame(pck_user_embeds)
        df_emb.insert(loc=0, column='user_id', value=pck_users)
        df_emb['label'] = ['baseModel'] * len(pck_users)

        return df_emb