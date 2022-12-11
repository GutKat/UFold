import _pickle as pickle
import sys
import os

from tqdm import tqdm
from datetime import date, datetime

import torch
import torch.optim as optim
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
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


def train(contact_net,train_generator,mcc_generator, validation_generator, epoches_first, lr):
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
    highest_mc = 0
    #Training
    print('start training... \n')
    # There are three steps of training
    # step one: train the u net
    # train for epoches in epoches_first, y is it called like that? y first?
    for epoch in range(epoches_first):
        contact_net.train()     #train on model
        for contacts, seq_embeddings, matrix_reps, seq_lens, seq_ori, seq_name, nc_map, l_len in tqdm(train_generator):
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
            #mcc_u = metrics.mcc(pred_contacts*contact_masks, contacts_batch)
            if write_tensorboard:
                writer.add_scalar(f"Loss/steps", loss_u, steps_done)
                #writer.add_scalar(f"MCC/steps", mcc_u, steps_done)
            # Optimize the model
            u_optimizer.zero_grad()
            loss_u.backward()
            u_optimizer.step()

            steps_done = steps_done+1

        #f1, prec, recall = metrics.model_eval_all_test(contact_net, train_generator)
        mcc_train = metrics.mcc_model(contact_net, mcc_generator, time_it=False, use_set=100)
        mcc_val = metrics.mcc_model(contact_net, validation_generator, time_it=False)

        #val_input, val_target = ...
        #val_output = contact_net(val_input)
        #val_loss = criterion(val_output, val_target)

        if write_tensorboard:
            #print to tensorboard in each epoch
            writer.add_scalar("Loss/train", loss_u, epoch)
            writer.add_scalar("MCC/train", mcc_train, epoch)
            writer.add_scalar('MCC/val', mcc_val, epoch)
            #writer.add_scalar('F1/train', f1, epoch)
            #writer.add_scalar('prec/train', prec, epoch)
            #writer.add_scalar('recall/train', recall, epoch)
            writer.flush()

        #print procress
        print('Training log: epoch: {}, step: {}, loss: {}, mcc: {}'.format(
                    epoch, steps_done-1, loss_u, mcc_train)) # f1: {}, prec: {}, recall: {}, f1, prec, recall
        print('Validation log: mcc: {}'.format(mcc_val))  # f1: {}, prec: {}, recall: {}, f1, prec, recall

        #save to folder
        if epoch > -1:
            if loss_u < lowest_loss:
                lowest_loss = loss_u
                best_model_loss = contact_net.state_dict()
            if mcc_train > highest_mc:
                highest_mc = highest_mc
                best_model_mcc = contact_net.state_dict()
            torch.save(contact_net.state_dict(),  f'ufold_training/{date_today}/{current_time}_{epoch}.pt')
    torch.save(best_model_loss, f'ufold_training/{date_today}/{current_time}_best_model_loss.pt')
    torch.save(best_model_mcc, f'ufold_training/{date_today}/{current_time}_best_model_mcc.pt')
    #print(save_best_model)
    contact_net.load_state_dict(best_model_loss)
    contact_net.to(device)
    f1, prec, recall = metrics.model_eval_all_test(contact_net, train_generator)
    mcc = metrics.mcc_model(contact_net, mcc_generator, time_it=False, use_set=100)
    print('Best model train: loss: {}, mcc: {}, f1: {}, prec: {}, recall: {}'.format(
        loss_u, mcc, f1, prec, recall))  # f1: {}, prec: {}, recall: {}, f1, prec, recall

    f1, prec, recall = metrics.model_eval_all_test(contact_net, validation_generator)
    mcc = metrics.mcc_model(contact_net, validation_generator, time_it=False)
    print('Best model validation: loss: {}, mcc: {}, f1: {}, prec: {}, recall: {}'.format(
        loss_u, mcc, f1, prec, recall))  # f1: {}, prec: {}, recall: {}, f1, prec, recall


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

    #I think this is just that we can use same config file for train and test
    #we do not use data_type or model_type here.
    data_type = config.data_type
    model_type = config.model_type
    lr = config.lr

    #epochs we want ot train for - dont get the name
    epoches_first = config.epoches_first
    train_files = args.train_files
    validation_file = r"random/pickle/N100_n160_val"

    # if gpu is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    seed_torch()

    params = {'batch_size': BATCH_SIZE,
              'shuffle': True,
              'num_workers': 4,
              'drop_last': False}

    params_evaluate = {'batch_size': 1}

    # for loading data
    # loading the rna ss data, the data has been preprocessed
    # train_data_list = []
    if len(train_files) != 1:
        train_data_list = []
        for file_item in train_files:
            print('Loading dataset:',file_item)
            if file_item == 'RNAStralign' or file_item == 'ArchiveII':
                train_data_list.append(RNASSDataGenerator('data/',file_item+'.pickle'))
            else:
                train_data_list.append(RNASSDataGenerator('data/',file_item+'.cPickle'))
            mcc_set = RNASSDataGenerator('data/',file_item+'.cPickle')
        train_merge = Dataset_FCN_merge(train_data_list)
        train_generator = data.DataLoader(train_merge, **params)
        # pdb.set_trace()
        mcc_set = Dataset_FCN(mcc_set)
        mcc_generator = data.DataLoader(mcc_set, **params_evaluate)
    else:
        print('Loading dataset:',train_files)
        file_item = train_files[0]
        if file_item == 'RNAStralign' or file_item == 'ArchiveII':
            train_set = RNASSDataGenerator('data/',file_item+'.pickle')
        else:
            train_set = RNASSDataGenerator('data/',file_item+'.cPickle')
        train_set = Dataset_FCN(train_set)
        train_generator = data.DataLoader(train_set, **params)
        # pdb.set_trace()
        mcc_generator = data.DataLoader(train_set, **params_evaluate)

    if file_item == 'RNAStralign' or validation_file == 'ArchiveII':
        validation_data = RNASSDataGenerator('data/', validation_file + '.pickle')
    else:
        validation_data = RNASSDataGenerator('data/', validation_file + '.cPickle')
    val_set = Dataset_FCN(validation_data)
    validation_generator = data.DataLoader(val_set, **params_evaluate)


    print('Data Loading Done!!!')
    import json
    if write_tensorboard:
        log_file = f"runs/log_files/log_{date_today}_{current_time}.log"
        with open(log_file, "w") as lf:
            lf.write(f"date: {date_today} \n")
            lf.write(f"time: {current_time} \n")
            lf.write(f"configuration: \n {json.dumps(config)}")

    # using the pytorch interface to parallel the data generation and model training


    
    contact_net = FCNNet(img_ch=17)
    if LOAD_MODEL:
        #/Users/katringutenbrunner/Desktop/UFold/ufold_training/11_12_2022
        MODEL_SAVED = r'ufold_training/11_12_2022/14_35_53.pt'.format(model_type, data_type, d)
        contact_net.load_state_dict(torch.load(MODEL_SAVED))
    contact_net.to(device)
    # for 5s
    # pos_weight = torch.Tensor([100]).to(device)
    # for length as 600

    #use train function to train the model
    train(contact_net, train_generator, mcc_generator, validation_generator, epoches_first, lr)

RNA_SS_data = collections.namedtuple('RNA_SS_data','seq ss_label length name pairs')


#model_eval_all_test()
if __name__ == '__main__':
    """
    See module-level docstring for a description of the script.
    """
    RNA_SS_data = collections.namedtuple('RNA_SS_data','seq ss_label length name pairs')
    write_tensorboard = True
    main()

#torch.save(contact_net.module.state_dict(), model_path + 'unet_final.pt')
# sys.exit()







