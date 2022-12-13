import _pickle as pickle
import sys
import os

import json
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


def train(contact_net,train_generator, validation_generator, test_generator, epoches_first, lr):
    # checking if the directory for models exist or not.
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
        pos_weight=pos_weight)

    # optimizer = Adam, without learning rate?
    u_optimizer = optim.Adam(contact_net.parameters(), lr=lr) #lr= 1e-4 #default = 1e-3

    # lowest loss and highest mcc for saving best model
    # lowest_loss = 10**10
    # highest_mcc = 0

    #Training
    print('start training... \n')
    # There are three steps of training
    # step one: train the u net
    # train for epoches in epoches_first, y is it called like that? y first?
    for epoch in range(epoches_first):
        contact_net.train()
        # for contacts, seq_embeddings, matrix_reps, seq_lens, seq_ori, seq_name in train_generator:
        for contacts, seq_embeddings, matrix_reps, seq_lens, seq_ori, seq_name in tqdm(train_generator):
            # for contacts, seq_embeddings, seq_embeddings_1, matrix_reps, seq_lens, seq_ori, seq_name in train_generator:
            # contacts, seq_embeddings, matrix_reps, seq_lens = next(iter(train_generator))

            contacts_batch = torch.Tensor(contacts.float()).to(device)
            seq_embedding_batch = torch.Tensor(seq_embeddings.float()).to(device)

            #get contact for prediction of model
            pred_contacts = contact_net(seq_embedding_batch)

            #mask out if seq is shorter than prediction , but it shoudln't be shorter than the prediction?
            contact_masks = torch.zeros_like(pred_contacts)
            for i in range(len(seq_lens)):
                contact_masks[i, :seq_lens[i], :seq_lens[i]] = 1

            #Compute loss
            loss_u = criterion_bce_weighted(pred_contacts*contact_masks, contacts_batch)
            #mcc_u = metrics.mcc(pred_contacts*contact_masks, contacts_batch)

            #write to tensoorboard
            if write_tensorboard:
                writer.add_scalar(f"Loss/steps", loss_u, steps_done)
                #writer.add_scalar(f"MCC/steps", mcc_u, steps_done)

            # Optimize the model
            u_optimizer.zero_grad()
            loss_u.backward()
            u_optimizer.step()
            steps_done = steps_done+1

        # calculate accuracy, etc. for each epoch
        #f1, prec, recall = metrics.model_eval_all_test(contact_net, train_generator)
        #mcc_train = metrics.mcc_model(contact_net, mcc_generator, time_it=False, use_set=50)
        mcc_val = metrics.mcc_model(contact_net, validation_generator, time_it=False)

        # print procress
        print('Training log: epoch: {}, step: {}, loss: {}'.format(
            epoch, steps_done - 1, loss_u))  # f1: {}, prec: {}, recall: {}, f1, prec, recall
        # print('Training log: epoch: {}, step: {}, loss: {}, mcc: {},  f1: {}, prec: {}, recall: {}'.format(
        #             epoch, steps_done-1, loss_u, mcc_train,  f1, prec, recall))
        print('training log: mcc: {}'.format(mcc_val))  # f1: {}, prec: {}, recall: {}, f1, prec, recall

        if write_tensorboard:
            #print to tensorboard in each epoch
            writer.add_scalar("Loss/train", loss_u, epoch)
            writer.add_scalar("MCC/val", mcc_val, epoch)
            writer.flush()

        #save to folder
        if epoch > -1:
            # if loss_u < lowest_loss:
            #     lowest_loss = loss_u
            #     best_model_loss = contact_net.state_dict()
            # if mcc_train > highest_mc:
            #     highest_mc = highest_mc
            #     best_model_mcc = contact_net.state_dict()
            torch.save(contact_net.state_dict(),  f'ufold_training/{date_today}/{current_time}_{epoch}.pt')
    #after training calculate accuracy for test set
    f1, prec, recall = metrics.model_eval_all_test(contact_net, test_generator)
    mcc_test = metrics.mcc_model(contact_net, test_generator)
    print('Test: mcc: {}, f1: {}, prec: {}, recall: {}'.format(mcc_test, f1, prec, recall))  # f1: {}, prec: {}, recall: {}, f1, prec, recall
    #test_generator
    # torch.save(best_model, f'ufold_training/{date_today}/{current_time}_best_model.pt')
    # #print(save_best_model)
    # contact_net.load_state_dict(best_model_loss)
    # contact_net.to(device)
    # f1, prec, recall = metrics.model_eval_all_test(contact_net, train_generator)
    # #mcc = metrics.mcc_model(contact_net, mcc_generator, time_it=False, use_set=50)
    # mcc = 0
    # print('Best model train: loss: {}, mcc: {}, f1: {}, prec: {}, recall: {}'.format(
    #     loss_u, mcc, f1, prec, recall))  # f1: {}, prec: {}, recall: {}, f1, prec, recall


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
    
    BATCH_SIZE = config.batch_size_stage_1
    lr = config.lr
    LOAD_MODEL = config.LOAD_MODEL
    #I think this is just that we can use same config file for train and test
    #we do not use data_type, model_type, d, OUT_STEP here.
    data_type = config.data_type
    model_type = config.model_type
    d = config.u_net_d
    #whats outstep
    OUT_STEP = config.OUT_STEP


    #epochs we want ot train for - dont get the name
    epoches_first = config.epoches_first
    train_files = args.train_files
    validation_file = r"random/pickle/N100_n200_val"
    test_file = r"random/pickle/N100_n200_test"

    # if gpu is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    seed_torch()


    # for loading data
    # loading the rna ss data, the data has been preprocessed
    # 5s data is just a demo data, which do not have pseudoknot, will generate another data having that
    train_data_list = []
    for file_item in train_files:
        print('Loading dataset: ', file_item)
        if file_item == 'RNAStralign' or file_item == 'ArchiveII':
            train_data_list.append(RNASSDataGenerator('data/', file_item + '.pickle'))
        else:
            train_data_list.append(RNASSDataGenerator('data/', file_item + '.cPickle'))

    print('Data Loading Done!!!')
    # using the pytorch interface to parallel the data generation and model training
    params = {'batch_size': BATCH_SIZE,
              'shuffle': True,
              'num_workers': 6,
              'drop_last': True}
    params_eval = {'batch_size': 1}

    #create generator for training, validating and testing
    train_merge = Dataset_FCN_merge(train_data_list)
    train_generator = data.DataLoader(train_merge, **params)


    validation_data = RNASSDataGenerator('data/', validation_file + '.cPickle')
    val_set = Dataset_FCN(validation_data)
    validation_generator = data.DataLoader(val_set, **params_eval)

    test_data = RNASSDataGenerator('data/', test_file + '.cPickle')
    test_set = Dataset_FCN(test_data)
    test_generator = data.DataLoader(test_set, **params_eval)


    print('Data Loading Done!!!')

    #if we write to tensorboard also create log_file with the config
    if write_tensorboard:
        log_file = f"runs/log_files/log_{date_today}_{current_time}.log"
        with open(log_file, "w") as lf:
            lf.write(f"date: {date_today} \n")
            lf.write(f"time: {current_time} \n")
            lf.write(f"configuration: \n {json.dumps(config)}")

    #our model
    contact_net = FCNNet(img_ch=17)

    if LOAD_MODEL:
        #/Users/katringutenbrunner/Desktop/UFold/ufold_training/11_12_2022
        MODEL_SAVED = "ufold_training/13_12_2022/14_26_1.pt"
        contact_net.load_state_dict(torch.load(MODEL_SAVED))
        print("Loaded Model!!")

    contact_net.to(device)
    # for 5s
    # pos_weight = torch.Tensor([100]).to(device)
    # for length as 600

    #use train function to train the model
    train(contact_net, train_generator, validation_generator, test_generator, epoches_first, lr)

RNA_SS_data = collections.namedtuple('RNA_SS_data','seq ss_label length name pairs')
if __name__ == '__main__':
    """
    See module-level docstring for a description of the script.
    """
    RNA_SS_data = collections.namedtuple('RNA_SS_data','seq ss_label length name pairs')
    write_tensorboard = True
    main()

#torch.save(contact_net.module.state_dict(), model_path + 'unet_final.pt')
# sys.exit()







