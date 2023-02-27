'''
Author: Katrin Gutenbrunner
script to predict structures from randomly created sequence and store predictions in numpy files
created files are used for the script ml_forensic.py
'''


from ufold_test import *
from torch.utils import data
from tqdm import tqdm
# from FCN import FCNNet
from Network import U_Net as FCNNet

from ufold.utils import *
from ufold.data_generator import RNASSDataGenerator, Dataset, RNASSDataGenerator_input
from ufold.data_generator import Dataset_Cut_concat_new as Dataset_FCN
import collections


args = get_args()
if args.nc:
    from ufold.postprocess import postprocess_new_nc as postprocess
else:
    from ufold.postprocess import postprocess_new as postprocess


def get_prediction_file(contact_net, test_generator,output_file):
    '''function to create numpy files with contact matrix from given network and given dataset

    Args:
        contact_net (Network.U_Net): UFold model, which should be tested
        test_generator (RNASSDataGenerator): generator with the dataset, which should be tested
        output_file (str): path under which the newly created files should be stored

    Returns:
        None
            creates two numpy files under output_file_matrix_nopp.npy and output_file_matrix_pp.npy
    '''
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


def main(folder, file_names, model = "models/ufold_train.pt", ):
    '''creates numpy files with predictions of input file with and without postprocessing

    Function creates two numpy files from the input file(s) (file_names), which are stored in the input folder.
    The two files are created using the function get_prediction_file and store the predictions of the model with and without postprocessing
    in the given folder.
    The files are used for the evaluations done in the file ml_forensic.py. A model can be specified with the
    model argument, as default the ufold model is selected

    Args:
        folder (str): path to the folder where the numpy files with the structure and sequences are saved, and where the nely created files are stores
        file_name (str, list): name of the file names which should be predicted, can be only one file as a string or multiple files as a list
        model (str): optional, path to the model, which should be used for the prediction, default = "models/ufold_train.pt"

    Returns:
        None
            creates the numpy files for the prediction with and without postprocessing

    '''
    torch.multiprocessing.set_sharing_strategy('file_system')
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    if type(file_names) == list:
        for file_name in file_names:
            if folder.endswith("/"):
                output_file = folder + file_name
            else:
                output_file = folder + "/" + file_name
            test_data = RNASSDataGenerator_input(folder, file_name)
            params = {'batch_size': 1,
                      'shuffle': False,
                      'num_workers': 6,
                      'drop_last': False}

            test_set = Dataset_FCN(test_data)
            test_generator = data.DataLoader(test_set, **params)
            contact_net = FCNNet(img_ch=17)

            print('==========Start Loading Pretrained Model==========')
            contact_net.load_state_dict(torch.load(model, map_location='cpu'))
            print(f"Model: {model} loaded")
            print('==========Finish Loading Pretrained Model==========')
            # contact_net = nn.DataParallel(contact_net, device_ids=[3, 4])
            contact_net.to(device)
            print(f'==========Start Predicting file {file_name}==========')
            get_prediction_file(contact_net, test_generator, output_file)
            print(f'==========Finish Predicting file {file_name}==========')
    else:
        file_name = file_names
        if folder.endswith("/"):
            output_file = folder + file_name
        else:
            output_file = folder + "/" + file_name
        test_data = RNASSDataGenerator_input(folder, file_name)
        params = {'batch_size': 1,
                  'shuffle': False,
                  'num_workers': 6,
                  'drop_last': False}

        test_set = Dataset_FCN(test_data)
        test_generator = data.DataLoader(test_set, **params)
        contact_net = FCNNet(img_ch=17)

        print('==========Start Loading Pretrained Model==========')
        contact_net.load_state_dict(torch.load(model, map_location='cpu'))
        print(f"Model: {model} loaded")
        print('==========Finish Loading Pretrained Model==========')
        # contact_net = nn.DataParallel(contact_net, device_ids=[3, 4])
        contact_net.to(device)
        print(f'==========Start Predicting file {file_name}==========')
        get_prediction_file(contact_net, test_generator, output_file)
        print(f'==========Finish Predicting file {file_name}==========')


if __name__ == '__main__':
    """
    See module-level docstring for a description of the script.
    """
    RNA_SS_data = collections.namedtuple('RNA_SS_data', 'seq ss_label length name pairs')
    folder = "data/analysis/type_analysis/ufold_model/"
    model = "models/ufold_train.pt"
    main(model = model, folder = folder)





