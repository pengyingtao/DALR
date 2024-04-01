'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2024-01-07 15:12:19
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2024-01-11 20:16:09
FilePath: /codes/LLMRec/DALR/encoder/train_encoder.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import sys
from config.configurator import configs
from trainer.trainer import init_seed

from data_utils.build_data_handler import build_data_handler
from models.bulid_model import build_model

from trainer.logger import Logger
from trainer.build_trainer import build_trainer
from trainer.tuner import Tuner


def main():
    # First Step: Create data_handler
    init_seed()

    ## load data 的处理过程
    data_handler = build_data_handler()
    data_handler.load_data()

    # Second Step: Create model
    model = build_model(data_handler).to(configs['device'])  ## 调用模型

    # Third Step: Create logger
    logger = Logger()

    # Fourth Step: Create trainer
    trainer = build_trainer(data_handler, logger)

    # Fifth Step: training
    trainer.train(model)

main()