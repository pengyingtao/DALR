'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2024-01-07 15:12:19
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2024-01-08 10:54:12
FilePath: /codes/LLMRec/DALR/encoder/models/bulid_model.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from config.configurator import configs
import importlib
import sys

def build_model(data_handler):
    model_type = configs['data']['type']
    model_name = configs['model']['name']
    module_path = ".".join(['models', model_type, model_name])  # models.general_cf.lightgcn

    if importlib.util.find_spec(module_path) is None:
        raise NotImplementedError('Model {} is not implemented'.format(model_name))
    
    module = importlib.import_module(module_path)  # 模块路径 DALR/encoder/models/general_cf/lightgcn.py
    
    for attr in dir(module):
        if attr.lower() == model_name.lower():
            '''getattr(module, attr)(data_handler)
            LightGCN(
              (edge_dropper): SpAdjEdgeDrop()
              )'''
            return getattr(module, attr)(data_handler)
    else:
        raise NotImplementedError('Model Class {} is not defined in {}'.format(model_name, module_path))
