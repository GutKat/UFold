
from torch.utils import data

from Network import U_Net as FCNNet
from ufold.utils import *
from ufold.config import process_config

from ufold.data_generator import RNASSDataGenerator
from ufold.data_generator import Dataset_Cut_concat_new_canonicle as Dataset_FCN
import collections
from ufold.ml_forensic import model_eval

args = get_args()



def main():
    torch.multiprocessing.set_sharing_strategy('file_system')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config_file = args.config
    n = 50
    N = 2000

    #test_file = f"random/length_test/N{N}_n{n}_test" # random/length_test/N100_n100_test
    test_file = f"random/length_test/n{n}/N{N}_n{n}_test"
    #test_file = "rnadeep/bpRNAinv120_valid_small"
    config = process_config(config_file)
    print('Here is the configuration of this run: ')
    print(config)
    models = [ "ufold_training/02_02_2023/17_59_0.pt", "ufold_training/02_02_2023/20_21_0.pt", "ufold_training/02_02_2023/21_24_0.pt", "ufold_training/02_02_2023/22_34_0.pt"]
    MODEL_SAVED = f"ufold_training/sorted/n{n}/N100000_e0.pt"
    #for MODEL_SAVED in models:
    #MODEL_SAVED = "ufold_training/04_01_2023/11_38_0.pt"
    #MODEL_SAVED = "ufold_training/04_01_2023/11_38_1.pt"
    #MODEL_SAVED = f"ufold_training/sorted/n{n}/N100000_e0.pt" #20_12_2022/15_17_0.pt
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed_torch()

    print('Loading test file: ',test_file)
    if test_file == 'RNAStralign' or test_file == 'ArchiveII':
        test_data = RNASSDataGenerator('data/', test_file+'.pickle')
    else:
        test_data = RNASSDataGenerator('data/',test_file+'.cPickle')

    # using the pytorch interface to parallel the data generation and model training
    params = {'batch_size': 1,
              'shuffle': True,
              'num_workers': 4,
              'drop_last': True}

    test_set = Dataset_FCN(test_data)
    test_generator = data.DataLoader(test_set, **params)

    contact_net = FCNNet(img_ch=17)

    print('==========Start Loading==========')
    print("Loaded Model:", MODEL_SAVED)
    contact_net.load_state_dict(torch.load(MODEL_SAVED, map_location=device))
    print('==========Finish Loading==========')
    contact_net.to(device)

    #evaluate the prediction of test set
    not_processed, processed = model_eval(contact_net, test_generator)
    print('No Postprocessing: MCC: {:1.2f}, f1: {:.2f}, prec: {:.2f}, recall: {:.2f}'.format(not_processed[0],not_processed[1],not_processed[2], not_processed[3]))
    print('Postprocessed: MCC: {:1.2f}, f1: {:1.2f}, prec: {:1.2f}, recall: {:1.2f}'.format(processed[0],processed[1], processed[2], processed[3]))

RNA_SS_data = collections.namedtuple('RNA_SS_data','seq ss_label length name pairs')
if __name__ == '__main__':
    RNA_SS_data = collections.namedtuple('RNA_SS_data','seq ss_label length name pairs')
    main()
