#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch

from models.data_utils import REGDataset
from models.model import REGShell
from models.config import Config

def main():
    # create instance of config
    config = Config(operation='evaluate')

    # build model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = REGShell(config, device)
    model.restore_model('results/train/20180905_112821/model/checkpoint.pth.tar')

    # create datasets
    test = REGDataset(config.filename_test, config=config, 
        max_iter=config.max_iter)
    
    # evaluate on test set
    model.evaluate(test)


if __name__ == "__main__":
    main()



