import os
import RNA
import numpy as np
import random
from ufold import utils
from ufold import random_generator
from matplotlib import pyplot as plt
from Network import U_Net as FCNNet
from ufold.utils import *
import collections
import re
from itertools import permutations, product
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=np.VisibleDeprecationWarning)
from ufold.data_generator import RNASSDataGenerator, Dataset
from ufold.data_generator import Dataset_Cut_concat_new_canonicle as Dataset_FCN
from torch.utils import data
from ufold.postprocess import postprocess_new as postprocess

utils.seed_torch(42)
random.seed(42)

perm = list(product(np.arange(4), np.arange(4)))
perm2 = [[1, 3], [3, 1]]
perm_nc = [[0, 0], [0, 2], [0, 3], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2], [3, 0], [3, 3]]


class random_input():
    def __init__(self, length):
        seq, ss, energy = random_generator.generate_random_seq_and_ss([length])[0]
        self.seq = seq
        self.ss = ss


def canon_bp(i, j):
    can_pair = {'A': {'A': 0, 'C': 0, 'G': 0, 'U': 1},
                'C': {'A': 0, 'C': 0, 'G': 1, 'U': 0},
                'G': {'A': 0, 'C': 1, 'G': 0, 'U': 1},
                'U': {'A': 1, 'C': 0, 'G': 1, 'U': 0}}
    return can_pair[i][j]


def base_pair_matrix(ss):
    # ptable[i] = j if (i.j) pair or 0 if i is unpaired,
    # ptable[0] contains the length of the structure.
    # ptable = RNA.ptable(ss)
    ptable = make_pair_table(ss, 1)
    matrix = np.zeros((len(ss), len(ss), 1), dtype=int)
    for i in range(1, len(ptable)):
        if ptable[i] != 0:
            j = ptable[i]
            matrix[i - 1][j - 1] = 1
    return matrix


def make_pair_table(ss, base=0, chars=['.']):
    stack = []
    if base == 0:
        pt = [-1] * len(ss)
    elif base == 1:
        pt = [0] * (len(ss) + base)
        pt[0] = len(ss)
    else:
        raise Exception(f"unexpected value in make_pair_table: (base = {base})")

    for i, char in enumerate(ss, base):
        if (char == '('):
            stack.append(i)
        elif (char == ')'):
            try:
                j = stack.pop()
            except IndexError as e:
                raise Exception("Too many closing brackets in secondary structure")
            pt[i] = j
            pt[j] = i
        elif (char not in set(chars)):
            raise Exception(f"unexpected character in sequence: '{char}'")
    if stack != []:
        raise Exception("Too many opening brackets in secondary structure")
    return pt


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    if type(a) == np.array:
        return np.allclose(a, a.T, rtol=rtol, atol=atol)
    elif type(a) == torch.Tensor:
        return torch.all(a.transpose(0, 1) == a)


def canonical_non_canonical(pairs, seq):
    if type(pairs) == str:
        ss = pairs
        ss = ss.replace("_", ".")
        pairs_sum = int((len(ss) - ss.count("."))/2)
        pairs = utils.ct2struct(ss)
    else:
        pairs_sum = len(pairs)
    canonical_pairs = 0
    for pair in pairs:
        canonical_pairs += canon_bp(seq[pair[0]], seq[pair[1]])
    non_canonical_pairs = pairs_sum - canonical_pairs
    return pairs_sum, canonical_pairs, non_canonical_pairs


def length_pairs(matrices, xlim=[0, 100], ylim=[0, 100]):
    results = []
    for matrix in matrices:
        results.append([int(np.sum(matrix) / 2), matrix.shape[0]])
    results = np.array(results)
    plt.scatter(results[:, 1], results[:, 0])
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1])
    plt.show()


char_dict = {
    0: 'A',
    1: 'U',
    2: 'C',
    3: 'G'
}
stats = {'S': 0,  # stack (actually: paired)
         'E': 0,  # exterior
         'B': 0,  # bulge
         'H': 0,  # hairpin
         'I': 0,  # interior
         'M': 0}  # multi
counts = {'#S': 0,  # stack (actually: paired)
          '#E': 0,  # exterior
          '#B': 0,  # bulge
          '#H': 0,  # hairpin
          '#I': 0,  # interior
          '#M': 0}  # multi


def looptypes(seqs, structs):
    stats = {'S': 0,  # stack (actually: paired)
             'E': 0,  # exterior
             'B': 0,  # bulge
             'H': 0,  # hairpin
             'I': 0,  # interior
             'M': 0}  # multi
    counts = {'#S': 0,  # stack (actually: paired)
              '#E': 0,  # exterior
              '#B': 0,  # bulge
              '#H': 0,  # hairpin
              '#I': 0,  # interior
              '#M': 0}  # multi
    for (seq, ss) in zip(seqs, structs):
        assert len(seq) == len(ss)
        # Counting paired vs unpaired is easy ...
        S = len([n for n in ss if n != '.'])
        L = len([n for n in ss if n == '.'])
        # ... but which loop are the unpaired ones in?
        tree = RNA.db_to_tree_string(ss, 5)  # type 5
        print(f'\r', end='')  # Unfortunately, the C function above prints a string!!!
        tdata = [x for x in tree.replace(")", "(").split("(") if x and x != 'R']
        scheck, lcheck = 0, 0
        for x in tdata:
            if x[0] == 'S':
                stats[x[0]] += 2 * int(x[1:]) / len(seq)
                counts[f'#{x[0]}'] += 1  # hmmm.... 1 or 2?
                scheck += 2 * int(x[1:])
            else:
                stats[x[0]] += int(x[1:]) / len(seq)
                counts[f'#{x[0]}'] += 1
                lcheck += int(x[1:])
        assert scheck == S and lcheck == L
    stats = {t: c / len(seqs) for t, c in stats.items()}
    counts = {t: c / len(seqs) for t, c in counts.items()}
    assert np.isclose(sum(stats.values()), 1.)
    return stats, counts


def get_data(filename):
    with open(filename, "r") as f:
        data = f.readlines()
    pattern = ">(.*)"
    test_data = []
    for i in range(0, len(data), 3):
        name = re.search(pattern, data[i])[1]
        seq = data[i + 1]
        seq = seq.replace("\n", "")
        ss = data[i + 2]
        ss = ss.replace("\n", "")
        test_data.append([name, seq, ss])
    return test_data


def get_cut_len(data_len, set_len):
    l = data_len
    if l <= set_len:
        l = set_len
    else:
        l = (((l - 1) // 16) + 1) * 16
    return l


def create_contact(data_seq, data_len, data_name):
    l = get_cut_len(data_len, 80)
    data_fcn = np.zeros((16, l, l))
    feature = np.zeros((8, l, l))
    if l >= 500:
        contact_adj = np.zeros((l, l))
        # contact_adj[:data_len, :data_len] = contact[:data_len, :data_len]
        # contact = contact_adj
        seq_adj = np.zeros((l, 4))
        seq_adj[:data_len] = data_seq[:data_len]
        data_seq = seq_adj
    for n, cord in enumerate(perm):
        i, j = cord
        data_fcn[n, :data_len, :data_len] = np.matmul(data_seq[:data_len, i].reshape(-1, 1),
                                                      data_seq[:data_len, j].reshape(1, -1))
    data_fcn_1 = np.zeros((1, l, l))
    data_fcn_1[0, :data_len, :data_len] = creatmat(data_seq[:data_len, ])
    data_fcn_2 = np.concatenate((data_fcn, data_fcn_1), axis=0)
    return data_fcn_2, data_len, data_seq[:l], data_name


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


def get_seq(contact):
    seq = None
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


def get_ct_dict_fast(predict_matrix, batch_num, seq_embedding, seq_name):
    seq_tmp = torch.mul(predict_matrix.cpu().argmax(axis=1),
                        predict_matrix.cpu().sum(axis=1).clamp_max(1)).numpy().astype(int)
    seq_tmpp = np.copy(seq_tmp)
    seq_tmp[predict_matrix.cpu().sum(axis=1) == 0] = -1

    dot_list = seq2dot((seq_tmp + 1).squeeze())
    letter = 'AUCG'
    seq_letter = ''.join([letter[item] for item in np.nonzero(seq_embedding)[:, 1]])

    seq = ((seq_tmp + 1).squeeze(), torch.arange(predict_matrix.shape[-1]).numpy() + 1)
    ct = [(seq[0][i] - 1, seq[1][i] - 1) for i in np.arange(len(seq[0])) if seq[0][i] != 0]
    dot = dot_list[:len(seq_letter)]
    # return ct_dict,dot_file_dict
    return ct, dot


def encoding2seq(arr):
    seq = list()
    for arr_row in list(arr):
        if sum(arr_row) == 0:
            seq.append('.')
        else:
            seq.append(char_dict[np.argmax(arr_row)])
    return ''.join(seq)


def contact2ct(contact, seq_len):
    contact = contact[:seq_len, :seq_len]
    structure = np.where(contact)
    pairs = []
    for i in range(len(structure[0])):
        if structure[0][i] < structure[1][i]:
            pairs.append((structure[0][i], structure[1][i]))
        else:
            pairs.append((structure[1][i], structure[0][i]))
    pairs = list(set(pairs))
    sec_struc = list("." * seq_len)
    for pair in pairs:
        sec_struc[pair[0]] = "("
        sec_struc[pair[1]] = ")"
    sec_struc = "".join(sec_struc)
    return pairs, sec_struc

def main():
    torch.multiprocessing.set_sharing_strategy('file_system')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_files = "analysis/random/N100_n100_test"  # random/length_test/N100_n100_test
    MODEL_SAVED = f"ufold_training/23_12_2022/12_49_0.pt"  # 20_12_2022/15_17_0.pt
    contact_net = FCNNet(img_ch=17)
    contact_net.load_state_dict(torch.load(MODEL_SAVED, map_location=device))
    contact_net.to(device)
    contact_net.train()
    test_data = RNASSDataGenerator('data/', test_files + '.cPickle')
    params = {'batch_size': 1,
              'shuffle': False,
              'num_workers': 6,
              'drop_last': True}
    test_set = Dataset_FCN(test_data)
    test_generator = data.DataLoader(test_set, **params)

    pairs_preds = []
    pairs_truths = []
    all_true_pred_no_pp = {}
    all_true_pred_pp = {}
    n = 1
    for contacts, seq_embeddings, matrix_reps, seq_lens, seq_ori, seq_name, nc_map, l_len in test_generator:
        seq = utils.encoding2seq(seq_ori[0].detach().numpy())[:seq_lens[0]]
        true_pairs, true_ss = contact2ct(contacts[0], seq_lens[0])

        seq_embedding_batch = torch.Tensor(seq_embeddings.float()).to(device)
        pred_contacts = contact_net(seq_embedding_batch)
        map_no_train = (pred_contacts > 0.5).float()
        map_no_train = map_no_train[0][:seq_lens[0], :seq_lens[0]]
        pred_pairs, pred_ss = get_ct_dict_fast(map_no_train, n, seq_ori.cpu().squeeze(), seq_name[0])
        pred_ss = pred_ss.replace("_", ".")
        pred_no_pp = {"pairs":pred_pairs, "ss": pred_ss}
        true = {"pairs":true_pairs, "ss": true_ss}
        all_true_pred_no_pp[seq_name[0]] = {"pred": pred_no_pp, "true": true, "seq":seq}

        u_no_train = postprocess(pred_contacts,seq_ori, 0.01, 0.1, 100, 1.6, True, 1.5)
        map_no_train = (u_no_train > 0.5).float()
        map_no_train = map_no_train[0][:seq_lens[0], :seq_lens[0]]
        pred_pairs, pred_ss = get_ct_dict_fast(map_no_train, n, seq_ori.cpu().squeeze(), seq_name[0])

        pred_ss = pred_ss.replace("_", ".")
        pred_pp = {"pairs": pred_pairs, "ss": pred_ss}
        all_true_pred_pp[seq_name[0]] = {"pred": pred_pp, "true": true, "seq":seq}
        n += 1

    bp_rel_true = []
    bp_rel_pred = []
    bp_diff = []
    canonical_bp_diff = []
    non_canonical_bp_diff = []

    true_bp = 0
    pred_bp = 0
    for i in all_true_pred_no_pp:
        seq = all_true_pred_no_pp[i]["seq"]
        pred_pairs = all_true_pred_no_pp[i]["pred"]["pairs"]
        true_pairs = all_true_pred_no_pp[i]["true"]["pairs"]

        pairs_pred = canonical_non_canonical(pred_pairs, seq)
        pairs_truth = canonical_non_canonical(true_pairs, seq)

        pred_bp += pairs_pred[0]
        true_bp += pairs_truth[0]
        bp_rel_true.append(pairs_truth[0]/len(seq))
        bp_rel_pred.append(pairs_pred[0]/len(seq))
        bp_diff.append(pairs_truth[0] - pairs_pred[0])
        canonical_bp_diff.append(pairs_truth[1] - pairs_pred[1])
        non_canonical_bp_diff.append(pairs_truth[2] - pairs_pred[2])
    print("without postprocessing")
    print(f"average true bp: {sum(bp_rel_true)/n:1.2f}")
    print(f"average pred bp: {sum(bp_rel_pred)/n:1.2f}")
    print(f"average total true bp: {true_bp / n:1.2f}")
    print(f"average total pred bp: {pred_bp / n:1.2f}")
    print(f"average bp diff: {sum(bp_diff)/n:1.2f}")
    print(f"average bp diff: {sum(canonical_bp_diff)/n:1.2f}")
    print(f"average bp diff: {sum(non_canonical_bp_diff)/n:1.2f}")

    bp_rel_pred = []
    bp_diff = []
    canonical_bp_diff = []
    non_canonical_bp_diff = []
    pred_bp = 0
    for i in all_true_pred_pp:
        seq = all_true_pred_pp[i]["seq"]
        pred_pairs = all_true_pred_pp[i]["pred"]["pairs"]
        true_pairs = all_true_pred_pp[i]["true"]["pairs"]
        pairs_pred = canonical_non_canonical(pred_pairs, seq)
        pairs_truth = canonical_non_canonical(true_pairs, seq)
        pred_bp += pairs_pred[0]

        bp_rel_pred.append(pairs_pred[0]/len(seq))
        bp_diff.append(pairs_truth[0] - pairs_pred[0])
        canonical_bp_diff.append(pairs_truth[1] - pairs_pred[1])
        non_canonical_bp_diff.append(pairs_truth[2] - pairs_pred[2])


    print("withpostprocessing")
    print(f"average rel true bp: {sum(bp_rel_true)/ n:1.2f}")
    print(f"average rel pred bp: {sum(bp_rel_pred) / n:1.2f}")
    print(f"average total true bp: {true_bp / n:1.2f}")
    print(f"average total pred bp: {pred_bp / n:1.2f}")
    print(f"average bp diff: {sum(bp_diff)/n:1.2f}")
    print(f"average bp diff: {sum(canonical_bp_diff)/n:1.2f}")
    print(f"average bp diff: {sum(non_canonical_bp_diff)/n:1.2f}")



RNA_SS_data = collections.namedtuple('RNA_SS_data', 'seq ss_label length name pairs')
if __name__ == '__main__':
    RNA_SS_data = collections.namedtuple('RNA_SS_data', 'seq ss_label length name pairs')
    main()
