'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2024-01-02 21:52:32
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2024-01-08 10:39:58
FilePath: /codes/LLMRec/RLMRec/encoder/data_utils/build_data_handler.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import sys
from config.configurator import configs
import importlib


def build_data_handler():

    datahandler_name = 'data_handler_' + configs['data']['type']
    # print(datahandler_name)  # data_handler_general_cf
    module_path = ".".join(['data_utils', datahandler_name])  ## 路径模块
    # print(module_path)  # data_utils.data_handler_general_cf

    if importlib.util.find_spec(module_path) is None:  ## 判断非空
        raise NotImplementedError('DataHandler {} is not implemented'.format(datahandler_name))
    
    module = importlib.import_module(module_path)
    
    for attr in dir(module):
        if attr.lower() == datahandler_name.lower().replace('_', ''):
            return getattr(module, attr)()
    else:
        raise NotImplementedError('DataHandler Class {} is not defined in {}'.format(datahandler_name, module_path))
