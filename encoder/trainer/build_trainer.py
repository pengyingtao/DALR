'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2024-01-02 21:52:32
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2024-01-08 11:02:20
FilePath: /codes/LLMRec/RLMRec/encoder/trainer/build_trainer.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from config.configurator import configs
import importlib
import sys

def build_trainer(data_handler, logger):
    trainer_name = 'Trainer' if 'trainer' not in configs['train'] else configs['train']['trainer']  
    print(trainer_name)  # 训练的参数
    # delete '_' in trainer name
    trainer_name = trainer_name.replace('_', '')

    trainers = importlib.import_module('trainer.trainer')  # DALR/encoder/trainer/trainer.py 这里是trainers的路径

    for attr in dir(trainers):
        if attr.lower() == trainer_name.lower():
            return getattr(trainers, attr)(data_handler, logger)  # 这里返回 trainer 类
    else:
        raise NotImplementedError('Trainer Class {} is not defined in {}'.format(trainer_name, 'trainer.trainer'))
