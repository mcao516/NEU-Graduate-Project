from models.data_utils import REGDataset
from models.reg_model import REGShell
from models.config import Config

def main():
    # create instance of config
    config = Config(operation='evaluate')

    # build model
    model = REGShell(config)
    model.restore_model('results/train/20180905_112821/model/checkpoint.pth.tar')

    # create datasets
    test = REGDataset(config.filename_test, config=config, 
        max_iter=config.max_iter)
    
    # evaluate on test set
    model.evaluate(test)


if __name__ == "__main__":
    main()



