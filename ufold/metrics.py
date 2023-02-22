import torch
import numpy as np
import _pickle as pickle
import sys
import os

import torch
import torch.optim as optim
from torch.utils import data

from Network import U_Net as FCNNet
from ufold.utils import *
import time
from ufold.postprocess import postprocess_new as postprocess
from tqdm import tqdm

def mcc(y_true, y_pred):
    y_pred = y_pred.clone().detach()
    y_true = y_true.clone().detach()
    y_pred_pos = torch.round(torch.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = torch.round(torch.clip(y_true, 0, 1))
    y_neg = 1 - y_pos
    tp = torch.sum(y_pos * y_pred_pos)
    tn = torch.sum(y_neg * y_pred_neg)
    fp = torch.sum(y_neg * y_pred_pos)
    fn = torch.sum(y_pos * y_pred_neg)
    numerator = (tp * tn - fp * fn)
    denominator = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    eps = 1e-10
    return numerator / (denominator + eps)


#
# def mcc_model(contact_net, test_generator, time_it=False):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     contact_net.train()
#     mcc_no_pp = 0
#     mcc_pp = 0
#     batch_n = 0
#     run_time = []
#
#     for contacts, seq_embeddings, matrix_reps, seq_lens, seq_ori, seq_name, nc_map, l_len in tqdm(test_generator):
#         tik = time.time()
#         if seq_lens.item() > 1500:
#             continue
#         batch_n += 1
#         contacts_batch = torch.Tensor(contacts.float()).to(device)
#         seq_embedding_batch = torch.Tensor(seq_embeddings.float()).to(device)
#
#         with torch.no_grad():
#             pred_contacts = contact_net(seq_embedding_batch)
#         mcc_no_pp += mcc(contacts_batch, pred_contacts)
#
#         #postprocessing
#         u_no_train = postprocess(pred_contacts,
#                                  seq_ori, 0.01, 0.1, 100, 1.6, True, 1.5)  ## 1.6
#         map_no_train = (u_no_train > 0.5).float()
#
#         tok = time.time()
#         t0 = tok - tik
#         run_time.append(t0)
#     mcc_no_pp /= batch_n
#     mcc_pp /= batch_n
#     if time_it:
#         print("sum run time:", np.sum(run_time))
#         print("mean run time:", np.mean(run_time))
#     return mcc_no_pp, mcc_pp



def model_eval(contact_net, test_generator):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    contact_net.train()
    result_no_pp_train = list()
    result_pp_train = list()
    mcc_no_pp = 0
    mcc_pp = 0
    batch_n = 0
    run_time = []

    for contacts, seq_embeddings, matrix_reps, seq_lens, seq_ori, seq_name, nc_map, l_len in tqdm(test_generator):
        batch_n += 1
        contacts_batch = torch.Tensor(contacts.float()).to(device)
        seq_embedding_batch = torch.Tensor(seq_embeddings.float()).to(device)

        #start run time measurement
        tik = time.time()
        with torch.no_grad():
            pred_contacts = contact_net(seq_embedding_batch)

        #get matrix with only 0 and 1 (threshold = 0.5)
        map_no_pp = (pred_contacts > 0.5).float()

        # post-processing
        pred_postprocessed = postprocess(pred_contacts, seq_ori, 0.01, 0.1, 100, 1.6, True, 1.5)
        # get matrix with only 0 and 1 (threshold = 0.5)
        map_pp = (pred_postprocessed > 0.5).float()

        #get run time of batch
        tok = time.time()
        t0 = tok - tik
        run_time.append(t0)

        #calculate mcc
        mcc_no_pp += mcc(contacts_batch, map_no_pp)
        mcc_pp += mcc(contacts_batch, map_pp)

        #calculate f1, precision and recall
        result_no_pp_tmp = list(map(lambda i: evaluate_exact_new(map_no_pp.cpu()[i], contacts_batch.cpu()[i]),range(contacts_batch.shape[0])))
        result_pp_tmp = list(map(lambda i: evaluate_exact_new(map_pp.cpu()[i], contacts_batch.cpu()[i]),range(contacts_batch.shape[0])))
        result_no_pp_train += result_no_pp_tmp
        result_pp_train += result_pp_tmp

    #divide mcc by size of dataset
    mcc_no_pp /= batch_n
    mcc_pp /= batch_n

    #unzip results (precision, recall and f1)
    p_no_pp, r_no_pp, f1_no_pp = zip(*result_no_pp_train)
    p_pp, r_pp, f1_pp = zip(*result_pp_train)
    return (mcc_no_pp, np.average(f1_no_pp), np.average(p_no_pp), np.average(r_no_pp)), (mcc_pp, np.average(f1_pp), np.average(p_pp), np.average(r_pp))