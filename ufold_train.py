import _pickle as pickle
import sys
import os

from tqdm import tqdm
from datetime import date, datetime

import torch
import torch.optim as optim
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
import pdb
import subprocess
from sklearn.metrics import matthews_corrcoef

# from FCN import FCNNet
from Network import U_Net as FCNNet

from ufold.utils import *
from ufold.config import process_config

from ufold.data_generator import RNASSDataGenerator, Dataset
from ufold.data_generator import Dataset_Cut_concat_new_merge_multi as Dataset_FCN_merge
from ufold.data_generator import Dataset_Cut_concat_new_canonicle as Dataset_FCN
import collections
import os

from ufold import metrics

date_today = date.today().strftime("%d_%m_%Y")
now = datetime.now()
current_time = now.strftime("%H_%M")


def train(contact_net,train_merge_generator, train_generator,epoches_first, lr):
    # checking if the directory for new training exist or not.
    if not os.path.exists(f"ufold_training/{date_today}"):
        os.makedirs(f"ufold_training/{date_today}")

    steps_done = 0
    if write_tensorboard:
        writer = SummaryWriter()
    #set epoch to zero
    epoch = 0

    #use cuda device if avaiable
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #why are we setting pos_weight?
    pos_weight = torch.Tensor([300]).to(device)
    criterion_bce_weighted = torch.nn.BCEWithLogitsLoss(
        pos_weight = pos_weight)

    #optimizer = Adam
    u_optimizer = optim.Adam(contact_net.parameters(), lr=lr)

    lowest_loss = 10**10
    #Training
    print('start training... \n')
    # There are three steps of training
    # step one: train the u net
    # train for epoches in epoches_first, y is it called like that? y first?
    for epoch in range(epoches_first):
        contact_net.train()     #train on model
        for contacts, seq_embeddings, matrix_reps, seq_lens, seq_ori, seq_name in tqdm(train_merge_generator):
            contacts_batch = torch.Tensor(contacts.float()).to(device)
            seq_embedding_batch = torch.Tensor(seq_embeddings.float()).to(device)
            #get contact for prediction of model
            pred_contacts = contact_net(seq_embedding_batch)

            #mask out if seq is shorter than prediction , but it shoudln't be shorter than the prediction?
            contact_masks = torch.zeros_like(pred_contacts)
            for i in range(len(seq_lens)):
                contact_masks[i, :seq_lens[i], :seq_lens[i]] = 1
            # Compute loss
            loss_u = criterion_bce_weighted(pred_contacts*contact_masks, contacts_batch)
            if write_tensorboard:
                writer.add_scalar(f"Loss/steps", loss_u, steps_done)
            # Optimize the model
            u_optimizer.zero_grad()
            loss_u.backward()
            u_optimizer.step()

            steps_done = steps_done+1

        #f1, prec, recall = metrics.model_eval_all_test(contact_net, train_generator)
        mcc = metrics.mcc_model(contact_net, train_generator)
        if write_tensorboard:
            #print to tensorboard in each epoch
            writer.add_scalar("Loss/train", loss_u, epoch)
            writer.add_scalar("MCC/train", mcc, epoch)
            #writer.add_scalar('F1/train', f1, epoch)
            #writer.add_scalar('prec/train', prec, epoch)
            #writer.add_scalar('recall/train', recall, epoch)
            writer.flush()

        #print procress
        print('Training log: epoch: {}, step: {}, loss: {}, mcc: {}'.format(
                    epoch, steps_done-1, loss_u, mcc)) # f1: {}, prec: {}, recall: {}, f1, prec, recall

        #save to folder
        if epoch > -1:
            if loss_u < lowest_loss:
                lowest_loss = loss_u
                save_best_model = contact_net.state_dict()
            torch.save(contact_net.state_dict(),  f'ufold_training/{date_today}_{current_time}_{epoch}.pt')
    torch.save(save_best_model, f'ufold_training/{date_today}_{current_time}_{lr}_best_model.pt')

def main():
    args = get_args()
    #cuda is commented, since my pc does not have a cuda
    #torch.cuda.set_device(1)

    #get configuration file from args
    config_file = args.config
    
    config = process_config(config_file)
    print("#####Stage 1#####")
    print('Here is the configuration of this run: ')
    print(config)

    os.environ["CUDA_VISIBLE_DEVICES"]= config.gpu
    
    d = config.u_net_d
    BATCH_SIZE = config.batch_size_stage_1
    #whats outstep
    OUT_STEP = config.OUT_STEP
    LOAD_MODEL = config.LOAD_MODEL

    #which data_types are possible? only pickle i think? which model_types are possible
    #I think this is just that we can use same config file for train and test
    #we do not use data_type or model_type here.
    data_type = config.data_type
    model_type = config.model_type
    lr = config.lr

    #epochs we want ot train for - dont get the name
    epoches_first = config.epoches_first
    train_files = args.train_files


    # if gpu is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    seed_torch()
    # for loading data
    # loading the rna ss data, the data has been preprocessed
    train_data_list = []
    for file_item in train_files:
        print('Loading dataset: ',file_item)
        if file_item == 'RNAStralign' or file_item == 'ArchiveII':
            train_data = RNASSDataGenerator('data/', file_item+ '.pickle')
            train_data_list.append(RNASSDataGenerator('data/',file_item+'.pickle'))
        else:
            train_data_list.append(RNASSDataGenerator('data/',file_item+'.cPickle'))
            train_data = RNASSDataGenerator('data/', file_item + '.cPickle')
    print('Data Loading Done!!!')
    import json
    if write_tensorboard:
        log_file = f"runs/log_files/log_{date_today}_{current_time}.log"
        with open(log_file, "w") as lf:
            lf.write(f"date: {date_today} \n")
            lf.write(f"time: {current_time} \n")
            lf.write(f"configuration: \n {json.dumps(config)}")


    # using the pytorch interface to parallel the data generation and model training
    params = {'batch_size': BATCH_SIZE,
              'shuffle': True,
              'num_workers': 4,
              'drop_last': False}

    params_evaluate = {'batch_size': 1,
              'shuffle': True,
              'num_workers': 4,
              'drop_last': True}

    train_merge = Dataset_FCN_merge(train_data_list)
    train_merge_generator = data.DataLoader(train_merge, **params)
    #pdb.set_trace()
    train_set = Dataset_FCN(train_data)
    train_generator = data.DataLoader(train_set, **params_evaluate)
    
    contact_net = FCNNet(img_ch=17)
    contact_net.to(device)

    # for 5s
    # pos_weight = torch.Tensor([100]).to(device)
    # for length as 600

    #use train function to train the model
    train(contact_net,train_merge_generator, train_generator,epoches_first, lr)

RNA_SS_data = collections.namedtuple('RNA_SS_data','seq ss_label length name pairs')


#model_eval_all_test()
if __name__ == '__main__':
    """
    See module-level docstring for a description of the script.
    """
    RNA_SS_data = collections.namedtuple('RNA_SS_data','seq ss_label length name pairs')
    write_tensorboard = False
    main()

#torch.save(contact_net.module.state_dict(), model_path + 'unet_final.pt')
# sys.exit()







