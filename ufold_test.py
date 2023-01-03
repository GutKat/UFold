import _pickle as pickle
import sys
import os

import torch
import torch.optim as optim
from torch.utils import data

from Network import U_Net as FCNNet
from ufold.utils import *
from ufold.config import process_config
import pdb
import time
from ufold.data_generator import RNASSDataGenerator, Dataset
from ufold.data_generator import Dataset_Cut_concat_new_canonicle as Dataset_FCN
import collections
from ufold.metrics import mcc_model, mcc_model_postprocessed, model_eval_all_test_no_postprocessing, model_eval_all_test_postprocessing


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


def get_ct_dict_fast(predict_matrix,batch_num,ct_dict,dot_file_dict,seq_embedding,seq_name):
    seq_tmp = torch.mul(predict_matrix.cpu().argmax(axis=1), predict_matrix.cpu().sum(axis = 1).clamp_max(1)).numpy().astype(int)
    seq_tmp[predict_matrix.cpu().sum(axis = 1) == 0] = -1
    dot_list = seq2dot((seq_tmp+1).squeeze())
    seq = ((seq_tmp+1).squeeze(),torch.arange(predict_matrix.shape[-1]).numpy()+1)
    letter='AUCG'
    ct_dict[batch_num] = [(seq[0][i],seq[1][i]) for i in np.arange(len(seq[0])) if seq[0][i] != 0]	
    seq_letter=''.join([letter[item] for item in np.nonzero(seq_embedding)[:,1]])
    dot_file_dict[batch_num] = [(seq_name,seq_letter,dot_list[:len(seq_letter)])]
    return ct_dict,dot_file_dict
# randomly select one sample from the test set and perform the evaluation


def model_eval_all_test(contact_net,test_generator):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    contact_net.train()
    pos_weight = torch.Tensor([300]).to(device)
    criterion_bce_weighted = torch.nn.BCEWithLogitsLoss(
        pos_weight = pos_weight)
    result_postprocessed = model_eval_all_test_no_postprocessing(contact_net, test_generator)
    result_no_postprocessing = model_eval_all_test_postprocessing(contact_net, test_generator)
    mcc_no_postprocess = mcc_model(contact_net, test_generator)
    mcc_postprocessed = mcc_model_postprocessed(contact_net, test_generator)
    #print(np.mean(run_time))
    print("MCC no postprocess: {:1.2f}, MCC postprocessed: {:1.2f}".format(mcc_no_postprocess, mcc_postprocessed))
    print('Postprocessed: f1: {:1.2f}, prec: {:1.2f}, recall: {:1.2f}'.format(result_postprocessed[0], result_postprocessed[1], result_postprocessed[2]))  # f1: {}, prec: {}, recall: {}, f1, prec, recall
    print('No Postprocessing: f1: {:.2f}, prec: {:.2f}, recall: {:.2f}'.format(result_no_postprocessing[0], result_no_postprocessing[1], result_no_postprocessing[2]))  # f1: {}, prec: {}, recall: {}, f1, prec, recall



    #with open('/data2/darren/experiment/ufold/results/sample_result.pickle','wb') as f:
    #    pickle.dump(result_dict,f)
    # with open('../results/rnastralign_short_pure_pp_evaluation_dict.pickle', 'wb') as f:
    #     pickle.dump(result_dict, f)


def main():
    torch.multiprocessing.set_sharing_strategy('file_system')
    #torch.cuda.set_device(2)
    #pdb.set_trace()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config_file = args.config
    #test_file = args.test_files
    test_file = r"random/length_test/N100_n100_test" # random/length_test/N100_n100_test

    config = process_config(config_file)
    print('Here is the configuration of this run: ')
    print(config)

    MODEL_SAVED = f"ufold_training/23_12_2022/12_49_0.pt" #20_12_2022/15_17_0.pt

    # if test_file not in ['TS1', 'TS2', 'TS3']:
    #     MODEL_SAVED = 'models/ufold_train.pt'
    # else:
    #     MODEL_SAVED = 'models/ufold_train_pdbfinetune.pt'

    # if gpu is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    seed_torch()
    
    # for loading data
    # loading the rna ss data, the data has been preprocessed
    # 5s data is just a demo data, which do not have pseudoknot, will generate another data having that

    print('Loading test file: ',test_file)
    if test_file == 'RNAStralign' or test_file == 'ArchiveII':
        test_data = RNASSDataGenerator('data/', test_file+'.pickle')
    else:
        test_data = RNASSDataGenerator('data/',test_file+'.cPickle')
    seq_len = test_data.data_y.shape[-2]

    print('Max seq length ', seq_len)
    #pdb.set_trace()
    
    # using the pytorch interface to parallel the data generation and model training
    params = {'batch_size': 1,
              'shuffle': True,
              'num_workers': 6,
              'drop_last': True}
    
    # test_set = Dataset(test_data)
    test_set = Dataset_FCN(test_data)
    test_generator = data.DataLoader(test_set, **params)
    
    '''
    test_merge = Dataset_FCN_merge(test_data,test_data2)
    test_merge_generator = data.DataLoader(test_merge, **params)
    pdb.set_trace()
    '''

    contact_net = FCNNet(img_ch=17)
    
    #pdb.set_trace()
    print('==========Start Loading==========')
    print("Loaded Model:", MODEL_SAVED)
    contact_net.load_state_dict(torch.load(MODEL_SAVED, map_location=device))
    #contact_net.load_state_dict(torch.load(MODEL_SAVED, map_location='cpu'))
    print('==========Finish Loading==========')
    # contact_net = nn.DataParallel(contact_net, device_ids=[3, 4])
    contact_net.to(device)
    model_eval_all_test(contact_net,test_generator)


RNA_SS_data = collections.namedtuple('RNA_SS_data','seq ss_label length name pairs')

if __name__ == '__main__':
    """
    See module-level docstring for a description of the script.
    """
    RNA_SS_data = collections.namedtuple('RNA_SS_data','seq ss_label length name pairs')
    main()




# Average testing F1 score with pure post-processing:  0.4097142
# Average testing precision with pure post-processing:  0.5006259
# Average testing recall with pure post-processing:  0.35211697