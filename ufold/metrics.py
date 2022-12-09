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
from sklearn.metrics import matthews_corrcoef
from ufold.config import process_config
import pdb
import time
from ufold.postprocess import postprocess_new as postprocess
from ufold.data_generator import RNASSDataGenerator, Dataset
from ufold.data_generator import Dataset_Cut_concat_new_canonicle as Dataset_FCN
import collections

# def mcc(y_true, y_pred):
#     y_pos = K.round(K.clip(y_true, 0, 1))
#     y_pred_pos = K.round(K.clip(y_pred, 0, 1))
#     neg_y_true = 1 - y_true
#     neg_y_pred = 1 - y_pred
#     tp = K.sum(y_true * y_pred)
#     fn = K.sum(y_true * neg_y_pred)
#     fp = K.sum(neg_y_true * y_pred)
#     tn = K.sum(neg_y_true * neg_y_pred)
#     numerator = (tp * tn - fp * fn)
#     denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
#     return numerator / (denominator + K.epsilon())


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
# def mcc_sklearn(contact_net, test_generator, time_it=False):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     contact_net.train()
#     mcc_list = 0
#     batch_n = 0
#     seq_names = []
#     seq_lens_list = []
#     run_time = []
#     pos_weight = torch.Tensor([300]).to(device)
#     criterion_bce_weighted = torch.nn.BCEWithLogitsLoss(
#         pos_weight=pos_weight)
#     for contacts, seq_embeddings, matrix_reps, seq_lens, seq_ori, seq_name, nc_map, l_len in test_generator:
#         tik = time.time()
#         nc_map_nc = nc_map.float() * contacts
#         if seq_lens.item() > 1500:
#             continue
#         batch_n += 1
#         contacts_batch = torch.Tensor(contacts.float()).to(device)
#         seq_embedding_batch = torch.Tensor(seq_embeddings.float()).to(device)
#         ##seq_embedding_batch_1 = torch.Tensor(seq_embeddings_1.float()).to(device)
#         with torch.no_grad():
#             pred_contacts = contact_net(seq_embedding_batch)
#         print(type(contacts_batch), type(pred_contacts))
#         mcc_list += matthews_corrcoef(contacts_batch, pred_contacts)
#
#         tok = time.time()
#         t0 = tok - tik
#         run_time.append(t0)
#     mcc_np = mcc_list / batch_n
#     if time_it:
#         print("all run times:", run_time)
#         print("sum run time in min :", np.sum(run_time) / 60)
#         print("mean run time:", np.mean(run_time))
#     return mcc_np

def mcc_model(contact_net, test_generator, time_it=False, use_set=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    contact_net.train()
    mcc_list = 0
    batch_n = 0
    run_time = []
    pos_weight = torch.Tensor([300]).to(device)
    criterion_bce_weighted = torch.nn.BCEWithLogitsLoss(
        pos_weight=pos_weight)
    if not use_set:
        for contacts, seq_embeddings, matrix_reps, seq_lens, seq_ori, seq_name, nc_map, l_len in test_generator:
            tik = time.time()
            if seq_lens.item() > 1500:
                continue
            batch_n += 1
            contacts_batch = torch.Tensor(contacts.float()).to(device)
            seq_embedding_batch = torch.Tensor(seq_embeddings.float()).to(device)
            ##seq_embedding_batch_1 = torch.Tensor(seq_embeddings_1.float()).to(device)
            with torch.no_grad():
                pred_contacts = contact_net(seq_embedding_batch)
            mcc_list += mcc(contacts_batch, pred_contacts)

            tok = time.time()
            t0 = tok - tik
            run_time.append(t0)
    else:
        for i in range(use_set):
            #contacts, seq_embeddings, matrix_reps, seq_lens, seq_ori, seq_name = next(iter(test_generator))
            contacts, seq_embeddings, matrix_reps, seq_lens, seq_ori, seq_name, nc_map, l_len = next(iter(test_generator))
            tik = time.time()
            if seq_lens.item() > 1500:
                continue
            batch_n += 1
            contacts_batch = torch.Tensor(contacts.float()).to(device)
            seq_embedding_batch = torch.Tensor(seq_embeddings.float()).to(device)
            with torch.no_grad():
                pred_contacts = contact_net(seq_embedding_batch)
            mcc_list += mcc(contacts_batch, pred_contacts)

            tok = time.time()
            t0 = tok - tik
            run_time.append(t0)
    mcc_np = mcc_list / batch_n
    if time_it:
        print("sum run time:", np.sum(run_time))
        print("mean run time:", np.mean(run_time))
    return mcc_np


def model_eval_all_test(contact_net, test_generator):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    contact_net.train()
    result_no_train = list()
    batch_n = 0
    result_nc = list()
    nc_name_list = []
    run_time = []
    pos_weight = torch.Tensor([300]).to(device)
    criterion_bce_weighted = torch.nn.BCEWithLogitsLoss(
        pos_weight=pos_weight)
    for contacts, seq_embeddings, matrix_reps, seq_lens, seq_ori, seq_name, nc_map, l_len in test_generator:
        # pdb.set_trace()
        nc_map_nc = nc_map.float() * contacts
        #if seq_lens.item() > 1500:
        #    continue
        batch_n += 1
        contacts_batch = torch.Tensor(contacts.float()).to(device)
        seq_embedding_batch = torch.Tensor(seq_embeddings.float()).to(device)
        ##seq_embedding_batch_1 = torch.Tensor(seq_embeddings_1.float()).to(device)
        tik = time.time()
        with torch.no_grad():
            pred_contacts = contact_net(seq_embedding_batch)

        # only post-processing without learning
        u_no_train = postprocess(pred_contacts,
                                 seq_ori, 0.01, 0.1, 100, 1.6, True, 1.5)  ## 1.6
        nc_no_train = nc_map.float().to(device) * u_no_train
        map_no_train = (u_no_train > 0.5).float()
        map_no_train_nc = (nc_no_train > 0.5).float()

        tok = time.time()
        t0 = tok - tik
        run_time.append(t0)

        result_no_train_tmp = list(map(lambda i: evaluate_exact_new(map_no_train.cpu()[i],
                                                                    contacts_batch.cpu()[i]),
                                       range(contacts_batch.shape[0])))
        result_no_train += result_no_train_tmp

        if nc_map_nc.sum() != 0:
            # pdb.set_trace()
            result_nc_tmp = list(map(lambda i: evaluate_exact_new(map_no_train_nc.cpu()[i],
                                                                  nc_map_nc.cpu().float()[i]),
                                     range(contacts_batch.shape[0])))
            result_nc += result_nc_tmp
            nc_name_list.append(seq_name[0])

    nt_exact_p, nt_exact_r, nt_exact_f1 = zip(*result_no_train)
    return np.average(nt_exact_f1), np.average(nt_exact_p), np.average(nt_exact_r)


    # with open('/data2/darren/experiment/ufold/results/sample_result.pickle','wb') as f:
    #    pickle.dump(result_dict,f)
    # with open('../results/rnastralign_short_pure_pp_evaluation_dict.pickle', 'wb') as f:
    #     pickle.dump(result_dict, f)

# def specificity(y_true, y_pred):
#     neg_y_true = 1 - y_true
#     neg_y_pred = 1 - y_pred
#     fp = K.sum(neg_y_true * y_pred)
#     tn = K.sum(neg_y_true * neg_y_pred)
#     specificity = tn / (tn + fp + K.epsilon())
#     return specificity
#
#
# def sensitivity(y_true, y_pred):
#     neg_y_true = 1 - y_true
#     neg_y_pred = 1 - y_pred
#     tp = K.sum(y_true * y_pred)
#     fn = K.sum(y_true * neg_y_pred)
#     sensitivity = tp / (tp + fn + K.epsilon())
#     return sensitivity
#
#
# def f1(y_true, y_pred):
#     neg_y_true = 1 - y_true
#     neg_y_pred = 1 - y_pred
#     tp = K.sum(y_true * y_pred)
#     fn = K.sum(y_true * neg_y_pred)
#     fp = K.sum(neg_y_true * y_pred)
#     tn = K.sum(neg_y_true * neg_y_pred)
#     sensitivity = tp / (tp + fn + K.epsilon())
#     precision = tp / (tp + fp + K.epsilon())
#     return (2 * ((sensitivity * precision) / (sensitivity + precision + K.epsilon())))
#
