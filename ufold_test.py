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


RNA_SS_data = collections.namedtuple('RNA_SS_data','seq ss_label length name pairs')
args = get_args()
if args.nc:
    from ufold.postprocess import postprocess_new_nc as postprocess
else:
    from ufold.postprocess import postprocess_new as postprocess


def get_seq(contact):
    seq = torch.mul(contact.argmax(axis=1), contact.sum(axis = 1).clamp_max(1))
    seq[contact.sum(axis = 1) == 0] = -1
    return seq


def seq2dot(seq):
    idx = np.arange(1, len(seq) + 1)
    dot_file = np.array(['_'] * len(seq))
    dot_file[seq > idx] = '('
    dot_file[seq < idx] = ')'
    dot_file[seq == 0] = '.'
    dot_file = ''.join(dot_file)
    return dot_file


def get_ct_dict(predict_matrix,batch_num,ct_dict):
    for i in range(0, predict_matrix.shape[1]):
        for j in range(0, predict_matrix.shape[1]):
            if predict_matrix[:,i,j] == 1:
                if batch_num in ct_dict.keys():
                    ct_dict[batch_num] = ct_dict[batch_num] + [(i,j)]
                else:
                    ct_dict[batch_num] = [(i,j)]
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
    result_no_train = list()
    result_no_train_shift = list()
    seq_lens_list = list()
    batch_n = 0
    result_nc = list()
    result_nc_tmp = list()
    ct_dict_all = {}
    dot_file_dict = {}
    seq_names = []
    nc_name_list = []
    seq_lens_list = []
    run_time = []
    pos_weight = torch.Tensor([300]).to(device)
    criterion_bce_weighted = torch.nn.BCEWithLogitsLoss(
        pos_weight = pos_weight)
    for contacts, seq_embeddings, matrix_reps, seq_lens, seq_ori, seq_name, nc_map, l_len in test_generator:
        #pdb.set_trace()
        nc_map_nc = nc_map.float() * contacts
        print(seq_lens.item())
        if seq_lens.item() > 1500:
            continue
        if batch_n%1000==0:
            print('Batch number: ', batch_n)
        batch_n += 1
        contacts_batch = torch.Tensor(contacts.float()).to(device)
        seq_embedding_batch = torch.Tensor(seq_embeddings.float()).to(device)
        ##seq_embedding_batch_1 = torch.Tensor(seq_embeddings_1.float()).to(device)
        seq_ori = torch.Tensor(seq_ori.float()).to(device)
        seq_names.append(seq_name[0])
        seq_lens_list.append(seq_lens.item())
        tik = time.time()
        
        with torch.no_grad():
            pred_contacts = contact_net(seq_embedding_batch)

        # only post-processing without learning
        u_no_train = postprocess(pred_contacts,
            seq_ori, 0.01, 0.1, 100, 1.6, True,1.5) ## 1.6
        nc_no_train = nc_map.float().to(device) * u_no_train
        map_no_train = (u_no_train > 0.5).float()
        map_no_train_nc = (nc_no_train > 0.5).float()
        
        tok = time.time()
        t0 = tok - tik
        run_time.append(t0)
        ## add fine tune
        #threshold = 0.5
        '''
        while map_no_train.sum(axis=1).max() > 1:
            u_no_train = postprocess(u_no_train,seq_ori, 0.01, 0.1, 50, 1.0, True)
            threshold += 0.005
            map_no_train = (u_no_train > threshold).float()
        '''
        ## end fine tune
        #pdb.set_trace()

        result_no_train_tmp = list(map(lambda i: evaluate_exact_new(map_no_train.cpu()[i],
            contacts_batch.cpu()[i]), range(contacts_batch.shape[0])))
        result_no_train += result_no_train_tmp
        
        if nc_map_nc.sum() != 0:
            #pdb.set_trace()
            result_nc_tmp = list(map(lambda i: evaluate_exact_new(map_no_train_nc.cpu()[i],
                nc_map_nc.cpu().float()[i]), range(contacts_batch.shape[0])))
            result_nc += result_nc_tmp
            nc_name_list.append(seq_name[0])

    #pdb.set_trace()
    #print(np.mean(run_time))

    #dot_ct_file = open('results/dot_ct_file.txt','w')
    nt_exact_p,nt_exact_r,nt_exact_f1 = zip(*result_no_train)
    #pdb.set_trace()
    print('Average testing F1 score with pure post-processing: ', np.average(nt_exact_f1))
    print('Average testing precision with pure post-processing: ', np.average(nt_exact_p))
    print('Average testing recall with pure post-processing: ', np.average(nt_exact_r))

    #with open('/data2/darren/experiment/ufold/results/sample_result.pickle','wb') as f:
    #    pickle.dump(result_dict,f)
    # with open('../results/rnastralign_short_pure_pp_evaluation_dict.pickle', 'wb') as f:
    #     pickle.dump(result_dict, f)


def main():
    torch.multiprocessing.set_sharing_strategy('file_system')
    #torch.cuda.set_device(2)
    #pdb.set_trace()
    
    config_file = args.config
    test_file = args.test_files
    test_file = "random/pickle/N100_n160_test"

    config = process_config(config_file)
    print('Here is the configuration of this run: ')
    print(config)

    # if test_file not in ['TS1','TS2','TS3']:
    #     MODEL_SAVED = 'models/ufold_train.pt'
    # else:
    #     MODEL_SAVED = 'models/ufold_train_pdbfinetune.pt'

    
    # os.environ["CUDA_VISIBLE_DEVICES"]= config.gpu
    #MODEL_SAVED = 'ufold_training/11_12_2022/19_13_best_model_loss.pt'
    MODEL_SAVED = 'ufold_training/11_12_2022/19_13_best_model_mcc.pt'
    #d (u_net_d) only for saving it in the right way (line 181)
    d = config.u_net_d
    BATCH_SIZE = config.batch_size_stage_1

    #still don't know what out step is
    OUT_STEP = config.OUT_STEP
    LOAD_MODEL = config.LOAD_MODEL

    #data_type and model_type only for saving it in the right way (line 181)
    data_type = config.data_type
    model_type = config.model_type
    model_path = r'/ufold_training/10_12_2022/17_03_best_model_loss.pt'.format(model_type, data_type,d)
    epoches_first = config.epoches_first
    
    
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
    params = {'batch_size': BATCH_SIZE,
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
    contact_net.load_state_dict(torch.load(MODEL_SAVED,map_location='cpu'))
    print('==========Finish Loading==========')
    # contact_net = nn.DataParallel(contact_net, device_ids=[3, 4])
    contact_net.to(device)
    model_eval_all_test(contact_net,test_generator)


if __name__ == '__main__':
    """
    See module-level docstring for a description of the script.
    """
    RNA_SS_data = collections.namedtuple('RNA_SS_data','seq ss_label length name pairs')
    main()






