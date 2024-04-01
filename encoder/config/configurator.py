'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2024-01-07 15:12:19
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2024-01-15 10:17:22
FilePath: /codes/LLMRec/DALR/encoder/config/configurator.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os
import yaml
import torch
import pickle
import argparse
import numpy as np
import torch.nn as nn

def parse_configure(model=None, dataset=None):
    parser = argparse.ArgumentParser(description='RLMRec')
    parser.add_argument('--model', type=str, default='LightGCN', help='Model name')
    parser.add_argument('--dataset', type=str, default='amazon', help='Dataset name')
    parser.add_argument('--device', type=str, default='cuda', help='cpu or cuda')
    parser.add_argument('--seed', type=int, default=None, help='Device number')
    parser.add_argument('--cuda', type=str, default='1', help='Device number')
    parser.add_argument('--noise', type=float, default='0.0', help='train data noise rate')
    parser.add_argument('--align_weight', type=float, default='0.0', help='align weight')

    args, _ = parser.parse_known_args()

    # cuda GPU
    if args.device == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

    # model name 转小写
    if model is not None:
        model_name = model.lower()
    elif args.model is not None:
        model_name = args.model.lower()
    else:
        model_name = 'default'
        # print("Read the default (blank) configuration.")

    # dataset 数据集
    if dataset is not None:
        args.dataset = dataset

    # find yml file
    if not os.path.exists('./encoder/config/modelconf/{}.yml'.format(model_name)):
        raise Exception("Please create the yaml file for your model first.")

    # read yml file
    with open('./encoder/config/modelconf/{}.yml'.format(model_name), encoding='utf-8') as f:
        config_data = f.read()
        configs = yaml.safe_load(config_data)
        configs['model']['name'] = configs['model']['name'].lower()

        if 'tune' not in configs:
            configs['tune'] = {'enable': False}
            
        configs['device'] = args.device
        configs['noise'] = args.noise
        configs['align_weight'] = args.align_weight

        if args.dataset is not None:
            configs['data']['name'] = args.dataset

        if args.seed is not None:
            configs['train']['seed'] = args.seed

        # semantic embeddings
        usrprf_embeds_path = "./data/{}/usr_emb_np.pkl".format(configs['data']['name'])
        itmprf_embeds_path = "./data/{}/itm_emb_np.pkl".format(configs['data']['name'])
        # usrprf_embeds_path = "./data/{}/usr_prf_gpt2_emb_np.pkl".format(configs['data']['name'])
        # itmprf_embeds_path = "./data/{}/itm_prf_gpt2_emb_np.pkl".format(configs['data']['name'])
        with open(usrprf_embeds_path, 'rb') as f:
            configs['usrprf_embeds'] = pickle.load(f)
        with open(itmprf_embeds_path, 'rb') as f:
            configs['itmprf_embeds'] = pickle.load(f)

        # user_update_embs_path = "./data/{}/usr_emb_upd.pkl".format(configs['data']['name'])
        # item_update_embs_path = "./data/{}/itm_emb_upd.pkl".format(configs['data']['name'])
        # user_update_embs_path = "./data/{}/usr_gpt2_emb_upd.pkl".format(configs['data']['name'])
        # item_update_embs_path = "./data/{}/itm_gpt2_emb_upd.pkl".format(configs['data']['name'])
        user_update_embs_path = "./data/{}/usr_merge_deepw_sample_emb.pkl".format(configs['data']['name'])
        item_update_embs_path = "./data/{}/itm_merge_deepw_sample_emb.pkl".format(configs['data']['name'])
        with open(user_update_embs_path, 'rb') as f:
            configs['user_update_embs'] = pickle.load(f)
        with open(item_update_embs_path, 'rb') as f:
            configs['item_update_embs'] = pickle.load(f)

        return configs

configs = parse_configure()
