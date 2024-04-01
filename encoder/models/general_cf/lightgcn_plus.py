import pickle
import torch
import torch as t
from torch import nn
from config.configurator import configs
from models.loss_utils import cal_bpr_loss, reg_params, cal_infonce_loss, cal_mim_loss
from models.base_model import BaseModel
from models.model_utils import SpAdjEdgeDrop
import sys
from sklearn.metrics import mutual_info_score
from sklearn.manifold import TSNE 
import pandas as pd
from configparser import *


init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class LightGCN_plus(BaseModel):
    def __init__(self, data_handler):
        super(LightGCN_plus, self).__init__(data_handler)
        self.adj = data_handler.torch_adj
        self.keep_rate = configs['model']['keep_rate']
        '''这里是初始化的uid 和 iid'''
        self.user_embeds = nn.Parameter(init(t.empty(self.user_num, self.embedding_size)))
        self.item_embeds = nn.Parameter(init(t.empty(self.item_num, self.embedding_size)))

        self.edge_dropper = SpAdjEdgeDrop()
        self.final_embeds = None
        self.is_training = False

        # hyper-parameter
        self.layer_num = self.hyper_config['layer_num']
        self.reg_weight = self.hyper_config['reg_weight']
        self.kd_weight = self.hyper_config['kd_weight']
        self.kd_temperature = self.hyper_config['kd_temperature']
        self.kd_temperature = self.hyper_config['kd_temperature']
        # self.deepw_temperature = self.hyper_config['deepw_temperature']
        self.align_weight = self.hyper_config['align_weight']

        # semantic-embeddings
        self.usrprf_embeds = t.tensor(configs['usrprf_embeds']).float().to(configs['device'])
        self.itmprf_embeds = t.tensor(configs['itmprf_embeds']).float().to(configs['device'])
        # mlp
        self.mlp = nn.Sequential(
            nn.Linear(self.usrprf_embeds.shape[1], (self.usrprf_embeds.shape[1] + self.embedding_size) // 2),
            nn.LeakyReLU(),
            nn.Linear((self.usrprf_embeds.shape[1] + self.embedding_size) // 2, self.embedding_size)
        )

        # '''这里是采用LLM表征的u_emb i_emb'''
        # self.user_embeds = nn.Parameter(self.mlp(self.usrprf_embeds.cpu()))
        # self.item_embeds = nn.Parameter(self.mlp(self.itmprf_embeds.cpu()))

        # infoNCE criterion
        self.criterion = nn.BCELoss(reduction='none')

        # semantic update embeddings
        self.user_update_embs = t.tensor(configs['user_update_embs']).float().to(configs['device'])
        self.item_update_embs = t.tensor(configs['item_update_embs']).float().to(configs['device'])
        # mlp_dw
        self.mlp_dw = nn.Sequential(
            nn.Linear(self.user_update_embs.shape[1], (self.user_update_embs.shape[1] + self.embedding_size) // 2),
            nn.LeakyReLU(),
            nn.Linear((self.user_update_embs.shape[1] + self.embedding_size) // 2, self.embedding_size)
        )

        '''以下定义我的改进层'''
        # Transformer编码器层
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=self.user_update_embs.shape[1],
            nhead=2,
            dim_feedforward= self.user_update_embs.shape[1] // 2
        )
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=2)

        self._init_weight()

    def _init_weight(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                init(m.weight)
    
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

        for i in range(self.layer_num):
            embeds = self._propagate(adj, embeds_list[-1])
            embeds_list.append(embeds)

        embeds = sum(embeds_list)
        self.final_embeds = embeds
        return embeds[:self.user_num], embeds[self.user_num:]


    def _pick_embeds(self, user_embeds, item_embeds, batch_data):
        ancs, poss, negs = batch_data
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]
        return anc_embeds, pos_embeds, neg_embeds


    def cal_loss(self, batch_data):
        self.is_training = True
        user_embeds, item_embeds = self.forward(self.adj, self.keep_rate)

        '''BPR loss'''
        anc_embeds, pos_embeds, neg_embeds = self._pick_embeds(user_embeds, item_embeds, batch_data)
        bpr_loss = cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds) / anc_embeds.shape[0]
        reg_loss = self.reg_weight * reg_params(self)

        '''kd loss'''  # 对比损失
        usrprf_embeds = self.mlp(self.usrprf_embeds)
        itmprf_embeds = self.mlp(self.itmprf_embeds)

        ancprf_embeds, posprf_embeds, negprf_embeds = self._pick_embeds(usrprf_embeds, itmprf_embeds, batch_data)
        kd_loss = cal_infonce_loss(anc_embeds, ancprf_embeds, usrprf_embeds, self.kd_temperature) + \
                  cal_infonce_loss(pos_embeds, posprf_embeds, posprf_embeds, self.kd_temperature) + \
                  cal_infonce_loss(neg_embeds, negprf_embeds, negprf_embeds, self.kd_temperature)
        kd_loss /= anc_embeds.shape[0]
        kd_loss *= self.kd_weight

        # loss = bpr_loss + reg_loss + kd_loss ## 注意这里增加了互信息对比学习的损失
        # losses = {'bpr_loss': bpr_loss, 'reg_loss': reg_loss, 'kd_loss': kd_loss}

        # 更新维度·
        # print(self.user_update_embs.shape)  # torch.Size([11000, 2304])
        self.user_update_embs = self.transformer_encoder(self.user_update_embs.unsqueeze(0))
        self.user_update_embs = self.user_update_embs.squeeze(0)
        self.item_update_embs = self.transformer_encoder(self.item_update_embs.unsqueeze(0))
        self.item_update_embs = self.item_update_embs.squeeze(0)
        # print(self.user_update_embs.shape)  # torch.Size([11000, 2304])

        usr_deepw_embeds = self.mlp_dw(self.user_update_embs)
        itm_deepw_embeds = self.mlp_dw(self.item_update_embs)  
        # print(usr_deepw_embeds.shape)  # torch.Size([11000, 32])

        '''alignment loss''' # 对齐损失
        anc_deepw_embeds, pos_deepw_embeds, neg_deepw_embeds = self._pick_embeds(usr_deepw_embeds, itm_deepw_embeds, batch_data)
        align_loss = cal_infonce_loss(anc_embeds, anc_deepw_embeds, usr_deepw_embeds, self.kd_temperature) + \
                     cal_infonce_loss(pos_embeds, pos_deepw_embeds, pos_deepw_embeds, self.kd_temperature) + \
                     cal_infonce_loss(neg_embeds, neg_deepw_embeds, neg_deepw_embeds, self.kd_temperature)
        align_loss /= anc_embeds.shape[0]
        align_loss *= self.align_weight

        # loss = bpr_loss + reg_loss + kd_loss + align_loss ## 注意这里增加了互信息对比学习的损失
        loss = bpr_loss + reg_loss + align_loss ## 注意这里增加了互信息对比学习的损失
        losses = {'bpr_loss': bpr_loss, 'reg_loss': reg_loss, 'kd_loss': kd_loss, 'align_loss': align_loss}

        # '''MIM loss'''
        # pos_score = torch.sigmoid(mutual_info_score(anc_embeds, anc_deepw_embeds))
        # user_mim_loss = self.criterion(pos_score, torch.ones_like(pos_score, dtype=torch.float32)).mean()
        # pos_score = torch.sigmoid(torch.mul(pos_embeds, pos_deepw_embeds).sum(dim=1)) 
        # item_pos_loss = self.criterion(pos_score, torch.ones_like(pos_score, dtype=torch.float32)).mean()
        # # neg_score = torch.sigmoid(torch.mul(neg_embeds, neg_deepw_embeds).sum(dim=1))
        # # item_neg_loss = self.criterion(neg_score, torch.ones_like(pos_score, dtype=torch.float32)).mean()
        # mim_loss = user_mim_loss + item_pos_loss
        # loss = bpr_loss + reg_loss + kd_loss + align_loss + mim_loss ## 注意这里增加了互信息对比学习的损失
        # losses = {'bpr_loss': bpr_loss, 'reg_loss': reg_loss, 'kd_loss': kd_loss, 'align_loss': align_loss, 'mim_loss': mim_loss}
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
        # df_emb['label'] = ['RLMRec'] * len(pck_users)
        df_emb['label'] = ['DALR'] * len(pck_users)

        return df_emb