import _pickle as pickle
import sys
import os

from datetime import date
import torch
import torch.optim as optim
from torch.utils import data

import pdb
import subprocess


date_today = date.today().strftime("%d_%m_%Y")
# from FCN import FCNNet
from Network import U_Net as FCNNet

from ufold.utils import *
from ufold.config import process_config

from ufold.data_generator import RNASSDataGenerator, Dataset
from ufold.data_generator import Dataset_Cut_concat_new_merge_multi as Dataset_FCN_merge
import collections


def train(contact_net,train_merge_generator,epoches_first):
    steps_done = 0

    #set epoch to zero
    epoch = 0

    #use cuda device if avaiable
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #why are we setting it weight?
    pos_weight = torch.Tensor([300]).to(device)
    criterion_bce_weighted = torch.nn.BCEWithLogitsLoss(
        pos_weight = pos_weight)

    #optimizer = Adam
    u_optimizer = optim.Adam(contact_net.parameters())

    #Training
    print('start training...')
    # There are three steps of training
    # step one: train the u net
    #epoch_rec = [] # maybe delete? not used
    #train for epoches in epoches_first, y is it called like that? y first?
    for epoch in range(epoches_first):
        contact_net.train()     #train on model

        for contacts, seq_embeddings, matrix_reps, seq_lens, seq_ori, seq_name in train_merge_generator:
            contacts_batch = torch.Tensor(contacts.float()).to(device)
            seq_embedding_batch = torch.Tensor(seq_embeddings.float()).to(device)

            #get contact for prediction of model
            pred_contacts = contact_net(seq_embedding_batch)

            #mask out if seq is shorter than prediction , but it shoudln't be shorter than the prediction?
            contact_masks = torch.zeros_like(pred_contacts)
            contact_masks[:, :seq_lens, :seq_lens] = 1
    
            # Compute loss
            loss_u = criterion_bce_weighted(pred_contacts*contact_masks, contacts_batch)

            # Optimize the model
            u_optimizer.zero_grad()
            loss_u.backward()
            u_optimizer.step()

            steps_done= steps_done+1

        #print procress
        print('Training log: epoch: {}, step: {}, loss: {}'.format(
                    epoch, steps_done-1, loss_u))
        #save to folder
        if epoch > -1:
            torch.save(contact_net.state_dict(),  f'ufold_training/{date_today}_{epoch}.pt')

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
    #which data_types are possible? only pickle i think?
    data_type = config.data_type
    # which model_types are possible
    model_type = config.model_type
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
            train_data_list.append(RNASSDataGenerator('data/',file_item+'.pickle'))
        else:
            train_data_list.append(RNASSDataGenerator('data/',file_item+'.cPickle'))
    print('Data Loading Done!!!')

    # using the pytorch interface to parallel the data generation and model training
    params = {'batch_size': BATCH_SIZE,
              'shuffle': True,
              'num_workers': 6,
              'drop_last': True}

    train_merge = Dataset_FCN_merge(train_data_list)
    train_merge_generator = data.DataLoader(train_merge, **params)
    #pdb.set_trace()
    
    contact_net = FCNNet(img_ch=17)
    contact_net.to(device)
    

    #if LOAD_MODEL and os.path.isfile(model_path):
    #    print('Loading u net model...')
    #    contact_net.load_state_dict(torch.load(model_path))

    # for 5s
    # pos_weight = torch.Tensor([100]).to(device)
    # for length as 600

    #use train function to train the model
    train(contact_net,train_merge_generator,epoches_first)

RNA_SS_data = collections.namedtuple('RNA_SS_data','seq ss_label length name pairs')


#model_eval_all_test()
if __name__ == '__main__':
    """
    See module-level docstring for a description of the script.
    """
    RNA_SS_data = collections.namedtuple('RNA_SS_data','seq ss_label length name pairs')
    main()

#torch.save(contact_net.module.state_dict(), model_path + 'unet_final.pt')
# sys.exit()







