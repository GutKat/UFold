import _pickle as pickle
import sys
import os

import torch
import torch.optim as optim
from torch.utils import data
from tqdm import tqdm
# from FCN import FCNNet
from Network import U_Net as FCNNet

from ufold.utils import *
from ufold.config import process_config
import pdb
import time
from ufold.data_generator import RNASSDataGenerator, Dataset, RNASSDataGenerator_input
from ufold.data_generator import Dataset_Cut_concat_new as Dataset_FCN
import collections

import subprocess

args = get_args()
if args.nc:
    from ufold.postprocess import postprocess_new_nc as postprocess
else:
    from ufold.postprocess import postprocess_new as postprocess


def get_seq(contact):
    seq = None
    seq = torch.mul(contact.argmax(axis=1), contact.sum(axis=1).clamp_max(1))
    seq[contact.sum(axis=1) == 0] = -1
    return seq


def seq2dot(seq):
    '''function to convert the sequence to dot-bracket notation.
    arguments:
        seq:
    return:
        dot_file: in dot bracket notation'''
    idx = np.arange(1, len(seq) + 1)
    dot_file = np.array(['_'] * len(seq))
    dot_file[seq > idx] = '('
    dot_file[seq < idx] = ')'
    dot_file[seq == 0] = '.'
    dot_file = ''.join(dot_file)
    return dot_file


def get_ct_dict(predict_matrix, batch_num, ct_dict):
    for i in range(0, predict_matrix.shape[1]):
        for j in range(0, predict_matrix.shape[1]):
            if predict_matrix[:, i, j] == 1:
                if batch_num in ct_dict.keys():
                    ct_dict[batch_num] = ct_dict[batch_num] + [(i, j)]
                else:
                    ct_dict[batch_num] = [(i, j)]
    return ct_dict


def get_ct_dict_fast(predict_matrix, batch_num, ct_dict, dot_file_dict, seq_embedding, seq_name):
    seq_tmp = torch.mul(predict_matrix.cpu().argmax(axis=1),
                        predict_matrix.cpu().sum(axis=1).clamp_max(1)).numpy().astype(int)
    seq_tmpp = np.copy(seq_tmp)
    seq_tmp[predict_matrix.cpu().sum(axis=1) == 0] = -1

    dot_list = seq2dot((seq_tmp + 1).squeeze())
    letter = 'AUCG'
    seq_letter = ''.join([letter[item] for item in np.nonzero(seq_embedding)[:, 1]])

    seq = ((seq_tmp + 1).squeeze(), torch.arange(predict_matrix.shape[-1]).numpy() + 1)
    ct_dict[batch_num] = [(seq[0][i], seq[1][i]) for i in np.arange(len(seq[0])) if seq[0][i] != 0]
    dot_file_dict[batch_num] = [(seq_name.replace('/', '_'), seq_letter, dot_list[:len(seq_letter)])]

    ct_file_output(ct_dict[batch_num], seq_letter, seq_name, 'results/save_ct_file')
    _, _, noncanonical_pairs = type_pairs(ct_dict[batch_num], seq_letter)
    tertiary_bp = [list(x) for x in set(tuple(x) for x in noncanonical_pairs)]
    str_tertiary = []

    for i, I in enumerate(tertiary_bp):
        if i == 0:
            str_tertiary += ('(' + str(I[0]) + ',' + str(I[1]) + '):color=""#FFFF00""')
        else:
            str_tertiary += (';(' + str(I[0]) + ',' + str(I[1]) + '):color=""#FFFF00""')

    tertiary_bp = ''.join(str_tertiary)

    # return ct_dict,dot_file_dict
    return ct_dict, dot_file_dict, tertiary_bp


def ct_file_output(pairs, seq, seq_name, save_result_path):
    # pdb.set_trace()
    col1 = np.arange(1, len(seq) + 1, 1)
    col2 = np.array([i for i in seq])
    col3 = np.arange(0, len(seq), 1)
    col4 = np.append(np.delete(col1, 0), [0])
    col5 = np.zeros(len(seq), dtype=int)

    for i, I in enumerate(pairs):
        col5[I[0] - 1] = int(I[1])
    col6 = np.arange(1, len(seq) + 1, 1)
    temp = np.vstack((np.char.mod('%d', col1), col2, np.char.mod('%d', col3), np.char.mod('%d', col4),
                      np.char.mod('%d', col5), np.char.mod('%d', col6))).T
    np.savetxt(os.path.join(save_result_path, seq_name.replace('/', '_')) + '.ct', (temp), delimiter='\t', fmt="%s",
               header='>seq length: ' + str(len(seq)) + '\t seq name: ' + seq_name.replace('/', '_'), comments='')

    return


def type_pairs(pairs, sequence):
    sequence = [i.upper() for i in sequence]

    AU_pair = []
    GC_pair = []
    GU_pair = []
    other_pairs = []
    for i in pairs:
        if [sequence[i[0] - 1], sequence[i[1] - 1]] in [["A", "U"], ["U", "A"]]:
            AU_pair.append(i)
        elif [sequence[i[0] - 1], sequence[i[1] - 1]] in [["G", "C"], ["C", "G"]]:
            GC_pair.append(i)
        elif [sequence[i[0] - 1], sequence[i[1] - 1]] in [["G", "U"], ["U", "G"]]:
            GU_pair.append(i)
        else:
            other_pairs.append(i)
    watson_pairs_t = AU_pair + GC_pair
    wobble_pairs_t = GU_pair
    other_pairs_t = other_pairs
    return watson_pairs_t, wobble_pairs_t, other_pairs_t


def get_prediction_file(contact_net, test_generator,output_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    contact_net.train()
    matrix_nopp_file = output_file + "_matrix_nopp.npy"
    matrix_pp_file = output_file + "_matrix_pp.npy"
    matrices_no_pp = []
    matrices_pp = []
    for seq_embeddings, seq_lens, seq_ori, seq_name in tqdm(test_generator):
        # for contacts, seq_embeddings, matrix_reps, seq_lens, seq_ori, seq_name, nc_map, l_len in test_generator:

        seq_embedding_batch = torch.Tensor(seq_embeddings.float()).to(device)
        seq_ori = torch.Tensor(seq_ori.float()).to(device)

        with torch.no_grad():
            pred_contacts = contact_net(seq_embedding_batch)

        map_no_pp = (pred_contacts > 0.5).float()
        map_no_pp = map_no_pp[0][:seq_lens[0], :seq_lens[0]]
        # only post-processing without learning
        u_no_train = postprocess(pred_contacts,
                                 seq_ori, 0.01, 0.1, 100, 1.6, True, 1.5)
        map_pp = (u_no_train > 0.5).float()
        map_pp = map_pp[0][:seq_lens[0], :seq_lens[0]]

        matrices_no_pp.append(map_no_pp)
        matrices_pp.append(map_pp)

    np.save(matrix_nopp_file, matrices_no_pp)
    np.save(matrix_pp_file, matrices_pp)


def main():
    torch.multiprocessing.set_sharing_strategy('file_system')
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    MODEL_SAVED = "ufold_training/06_01_2023/12_11_2.pt"
    root_folder = "data/analysis/length/length_inverse/"
    for n in range(30,251,10):
        test_file = f"N1000_n{n}_ss"
        output_file = f"data/analysis/length/length_inverse/N1000_n{n}"
        test_data = RNASSDataGenerator_input(root_folder, test_file)
        params = {'batch_size': 1,
                  'shuffle': False,
                  'num_workers': 6,
                  'drop_last': False}

        test_set = Dataset_FCN(test_data)
        test_generator = data.DataLoader(test_set, **params)
        contact_net = FCNNet(img_ch=17)

        print('==========Start Loading Pretrained Model==========')
        contact_net.load_state_dict(torch.load(MODEL_SAVED, map_location='cpu'))
        print(f"Model: {MODEL_SAVED} loaded")
        print('==========Finish Loading Pretrained Model==========')
        # contact_net = nn.DataParallel(contact_net, device_ids=[3, 4])
        contact_net.to(device)
        print(f'==========Start Predicting file {test_file}==========')
        get_prediction_file(contact_net, test_generator, output_file)
        print(f'==========Finish Predicting file {test_file}==========')


if __name__ == '__main__':
    """
    See module-level docstring for a description of the script.
    """
    RNA_SS_data = collections.namedtuple('RNA_SS_data', 'seq ss_label length name pairs')
    main()





