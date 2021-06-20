import logging
import os

import numpy as np
import torch
from preparedata import partition_data,get_dataloader

def client_data(args_datadir="../data",client_number=10,batch_size=64,selection=None,args_alpha=0.5,):
    torch.manual_seed(1)
    logging.info("load dataset")
    logging.info(selection)
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(args_datadir,
                                                                                             client_number,
                                                                                             args_alpha,
                                                                                             selection)
    logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    all_train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])

    train_datas=[]
    for i in range(client_number):
        dataidxs = net_dataidx_map[i]
        local_sample_number = len(dataidxs)
        logging.info("client %d has %d sample"%(i,local_sample_number))
        #split = int(np.floor(0.5 * local_sample_number))  # split index
        #train_idxs = dataidxs[0:split]
        #test_idxs = dataidxs[split:local_sample_number]

        train_local, _ = get_dataloader(args_datadir, batch_size, dataidxs)
        logging.info("batch_num_train_local = %d" % (len(train_local)))
        train_datas.append(train_local)

        #test_local, _ = get_dataloader(args.dataset, args_datadir, args.batch_size, args.batch_size, test_idxs)
        #logging.info("rank = %d, batch_num_test_local = %d" % (rank, len(test_local)))

    #print(train_datas)
    return train_datas
    
#if __name__ == "__main__":
#    client_data()
