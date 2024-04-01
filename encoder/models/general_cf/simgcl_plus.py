import pickle
import sys
import torch
import torch as t
from torch import nn
import torch.nn.functional as F
from config.configurator import configs
from models.general_cf.lightgcn import LightGCN
from models.loss_utils import cal_bpr_loss, reg_params, cal_infonce_loss
from sklearn.manifold import TSNE 
import pandas as pd



init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class SimGCL_plus(LightGCN):
    def __init__(self, data_handler):
        super(SimGCL_plus, self).__init__(data_handler)

        # hyper-parameter
        self.cl_weight = self.hyper_config['cl_weight']
        self.cl_temperature = self.hyper_config['cl_temperature']
        self.kd_weight = self.hyper_config['kd_weight']
        self.kd_temperature = self.hyper_config['kd_temperature']
        self.eps = self.hyper_config['eps']
        # self.align_weight = self.hyper_config['align_weight']
        self.align_weight = configs['align_weight']

        # semantic-embedding
        self.usrprf_embeds = t.tensor(configs['usrprf_embeds']).float().cuda()
        self.itmprf_embeds = t.tensor(configs['itmprf_embeds']).float().cuda()
        self.mlp = nn.Sequential(
            nn.Linear(self.usrprf_embeds.shape[1], (self.usrprf_embeds.shape[1] + self.embedding_size) // 2),
            nn.LeakyReLU(),
            nn.Linear((self.usrprf_embeds.shape[1] + self.embedding_size) // 2, self.embedding_size)
        )

        '''这里是采用LLM表征的u_emb i_emb'''
        self.user_embeds = nn.Parameter(self.mlp(self.usrprf_embeds.cpu()))
        self.item_embeds = nn.Parameter(self.mlp(self.itmprf_embeds.cpu()))

        # semantic update embeddings
        self.user_update_embs = t.tensor(configs['user_update_embs']).float().cuda()
        self.item_update_embs = t.tensor(configs['item_update_embs']).float().cuda()
        self.mlp_dw = nn.Sequential(
            nn.Linear(self.user_update_embs.shape[1], (self.user_update_embs.shape[1] + self.embedding_size) // 2),
            nn.LeakyReLU(),
            nn.Linear((self.user_update_embs.shape[1] + self.embedding_size) // 2, self.embedding_size)
        )

        self._init_weight()


    def _init_weight(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                init(m.weight)

    def _perturb_embedding(self, embeds):
        noise = (F.normalize(t.rand(embeds.shape).cuda(), p=2) * t.sign(embeds)) * self.eps
        return embeds + noise
    
    def forward(self, adj=None, perturb=False):
        if adj is None:
            adj = self.adj
        if not perturb:
            return super(SimGCL_plus, self).forward(adj, 1.0)
        
        embeds = t.concat([self.user_embeds, self.item_embeds], dim=0)
        embeds_list = [embeds]
        for i in range(self.layer_num):
            embeds = self._propagate(adj, embeds_list[-1])
            embeds = self._perturb_embedding(embeds)
            embeds_list.append(embeds)
        embeds = sum(embeds_list)

        return embeds[:self.user_num], embeds[self.user_num:]
    
    
    def _pick_embeds(self, user_embeds, item_embeds, batch_data):
        ancs, poss, negs = batch_data
        anc_embeds = user_embeds[ancs]
        pos_embeds = item_embeds[poss]
        neg_embeds = item_embeds[negs]
        return anc_embeds, pos_embeds, neg_embeds
        
    def cal_loss(self, batch_data):
        self.is_training = True
        user_embeds1, item_embeds1 = self.forward(self.adj, perturb=True)
        user_embeds2, item_embeds2 = self.forward(self.adj, perturb=True)
        user_embeds3, item_embeds3 = self.forward(self.adj, perturb=False)

        anc_embeds1, pos_embeds1, neg_embeds1 = self._pick_embeds(user_embeds1, item_embeds1, batch_data)
        anc_embeds2, pos_embeds2, neg_embeds2 = self._pick_embeds(user_embeds2, item_embeds2, batch_data)
        anc_embeds3, pos_embeds3, neg_embeds3 = self._pick_embeds(user_embeds3, item_embeds3, batch_data)

        usrprf_embeds = self.mlp(self.usrprf_embeds)
        itmprf_embeds = self.mlp(self.itmprf_embeds)
        ancprf_embeds, posprf_embeds, negprf_embeds = self._pick_embeds(usrprf_embeds, itmprf_embeds, batch_data)

        bpr_loss = cal_bpr_loss(anc_embeds3, pos_embeds3, neg_embeds3) / anc_embeds3.shape[0]
        reg_loss = self.reg_weight * reg_params(self)

        cl_loss = cal_infonce_loss(anc_embeds1, anc_embeds2, user_embeds2, self.cl_temperature) + \
                  cal_infonce_loss(pos_embeds1, pos_embeds2, item_embeds2, self.cl_temperature)
        cl_loss /= anc_embeds1.shape[0]
        cl_loss *= self.cl_weight

        kd_loss = cal_infonce_loss(anc_embeds3, ancprf_embeds, usrprf_embeds, self.kd_temperature) + \
                  cal_infonce_loss(pos_embeds3, posprf_embeds, posprf_embeds, self.kd_temperature) + \
                  cal_infonce_loss(neg_embeds3, negprf_embeds, negprf_embeds, self.kd_temperature)
        kd_loss /= anc_embeds3.shape[0]
        kd_loss *= self.kd_weight

        # loss = bpr_loss + reg_loss + cl_loss + kd_loss
        # losses = {'bpr_loss': bpr_loss, 'reg_loss': reg_loss, 'cl_loss': cl_loss, 'kd_loss': kd_loss}

        '''alignment loss''' # 对齐损失
        usr_deepw_embeds = self.mlp_dw(self.user_update_embs)
        itm_deepw_embeds = self.mlp_dw(self.item_update_embs)
        anc_deepw_embeds, pos_deepw_embeds, neg_deepw_embeds = self._pick_embeds(usr_deepw_embeds, itm_deepw_embeds, batch_data)
        align_loss = cal_infonce_loss(anc_embeds3, anc_deepw_embeds, usr_deepw_embeds, self.kd_temperature) + \
                     cal_infonce_loss(pos_embeds3, pos_deepw_embeds, pos_deepw_embeds, self.kd_temperature) # + \
                    #  cal_infonce_loss(neg_embeds3, neg_deepw_embeds, neg_deepw_embeds, self.kd_temperature)
        # print(align_loss)  # 106338.8203
        align_loss /= anc_embeds3.shape[0]
        align_loss *= self.align_weight

        loss = bpr_loss + reg_loss + cl_loss + kd_loss + align_loss
        losses = {'bpr_loss': bpr_loss, 'reg_loss': reg_loss, 'cl_loss': cl_loss, 'kd_loss': kd_loss, 'align_loss': align_loss}

        return loss, losses
    

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
    

    def full_predict(self, batch_data):
        user_embeds, item_embeds = self.forward(self.adj, False)
        self.is_training = False
        pck_users, train_mask = batch_data
        pck_users = pck_users.long()
        pck_user_embeds = user_embeds[pck_users]
        full_preds = pck_user_embeds @ item_embeds.T  # numpy.dot（）的作用是一样的，矩阵乘法
        full_preds = self._mask_predict(full_preds, train_mask)
        return full_preds