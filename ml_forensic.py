'''
Based on the ml_fornesic.py of RNADeep
Expanded for the Softwareproject 2022/23
Contains functions to analysis secondary structure predicitons
'''

import os
import RNA
import numpy as np
from itertools import combinations
from matplotlib import pyplot as plt
import torch
from ufold.utils import *
import time
from ufold.postprocess import postprocess_new as postprocess
from tqdm import tqdm


def julia_version(a):
    # make symmetric
    a = (a + a.T) / 2

    # Take the maximum from each row and write it into a vector
    row_maxes = a.max(axis=1).reshape(-1, 1)
    col_maxes = a.max(axis=0).reshape(-1, 1)
    assert np.allclose(row_maxes, col_maxes)

    # Translate to only 0's and 1's dependent on th >= 0.5
    return np.where((a == row_maxes) & (a >= 0.5), 1, 0)


def remove_conflicts(a, seq=None):
    """
    """
    # extract make symmteric upper triangle
    a = np.triu((a + a.T) / 2, 1)
    # remove unpaired elements
    a = np.where((a < 0.5), 0, a)

    nbp = np.count_nonzero(a)

    # Get indices of the largest element in a
    (i, j) = np.unravel_index(a.argmax(), a.shape)
    while (i, j) != (0, 0):
        if not seq or canon_bp(seq[i], seq[j]):
            a[i,], a[:, j] = 0, 0  # looks inefficient
            a[j,], a[:, i] = 0, 0  # looks inefficient
            a[i, j] = -1
        else:
            a[i, j] = 0
        (i, j) = np.unravel_index(a.argmax(), a.shape)
    return np.where((a == -1), 1, a), nbp


def canon_bp(i, j):
    can_pair = {'A': {'A': 0, 'C': 0, 'G': 0, 'U': 1},
                'C': {'A': 0, 'C': 0, 'G': 1, 'U': 0},
                'G': {'A': 0, 'C': 1, 'G': 0, 'U': 1},
                'U': {'A': 1, 'C': 0, 'G': 1, 'U': 0}}
    return can_pair[i][j]


def julia_prediction(seqs, data):
    nn_structs = []
    collisions = 0
    tot_nbp = 0
    for (seq, nnd) in zip(seqs, data):
        ## remove one nesting
        a = np.reshape(nnd, (len(seq), len(seq)))

        # version 1: allow all base-pairs
        a, nbp = remove_conflicts(a)
        # tot_nbp += nbp
        ## version 2: only canonical base-pairs
        # a = remove_conflicts(a, seq)
        # version 3: juila's version (with symmetry correction)
        # a = julia_version(a)
        # unique, counts = np.unique(a, return_counts=True)

        # Make a pair table by looping over the upper triangular matrix ...
        pt = np.zeros(len(seq) + 1, dtype=int)
        pt[0] = len(seq)
        for i in range(len(seq)):
            for j in range(i + 1, len(seq)):
                if a[i][j] == 1:
                    if pt[i + 1] == pt[j + 1] == 0:
                        pt[i + 1], pt[j + 1] = j + 1, i + 1
                    else:
                        collisions += 1

        # remove pseudoknots & convert to dot-bracket
        ptable = tuple(map(int, pt))
        processed = RNA.pt_pk_remove(ptable)
        nns = RNA.db_from_ptable(processed)
        nn_structs.append(nns)
    tot_nbp /= len(seqs)
    # print(tot_nbp)
    # print(f'{collisions = }')
    return nn_structs


def julia_looptypes(seqs, structs):
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


def get_bp_counts(seqs, structs):
    counts = {}
    for (seq, ss) in zip(seqs, structs):
        pt = RNA.ptable(ss)
        for i, j in enumerate(pt[1:], 1):
            if j == 0 or i > j:
                continue
            bp = (seq[i - 1], seq[j - 1])
            counts[bp] = counts.get(bp, 0) + 1
    return counts


def length_analysis(folder, n_lengths, stem_file_name = "", save = False):
    '''
    function to analysis the relation between number of bp and length of sequences
     of real data and of given prediction without postprocessing and with postprocessing and create corresponding plot
     the files in the given folder should have structure:
        - folder/stem_file_name_n_sequence.npy
        - folder/stem_file_name_n_structure.npy
        - folder/stem_file_name_n_matrix_nopp.npy
        - folder/stem_file_name_n_matrix_pp.npy
    for each n in n_lengths
    arguments:
        folder: str, path to the folder where the required files are stored
            files needed: sequence.npy, structure.npy, matrix_nopp.npy, matrix_pp.npy)
            [npy files created by metrics.random_ml_forensic or metrics.fa2npy, npy files created by predict.py]
        n_lengths: list, list with the length of sequences (n), which should be evaluated
        stem_file_name: str, stem name of the prediction file, e.g. for 1000 sequences stem_file_name = "N1000"
        save: Boolean or str, should the created plot be saved, if False no plot is saved, if str: plot is saved under this path
    return:
        None
        plots the plot of bp vs length
    '''
    #length we want to test
    bp_count_truth = []
    bp_count_no_pp = []
    bp_count_pp = []
    ids_nopp = []
    ids_pp = []
    if not folder.endswith("/"):
        folder += "/"
    for n in tqdm(n_lengths):
        prediction_file = f"{folder}{stem_file_name}_n{n}"

        sequence_file = prediction_file + "_sequence.npy"
        structure_file = prediction_file + "_structure.npy"
        matrix_nopp_file = prediction_file + "_matrix_nopp.npy"
        matrix_pp_file = prediction_file + "_matrix_pp.npy"

        # load the files
        seqs = np.load(sequence_file)
        vrna = np.load(structure_file)
        data_no_pp = np.load(matrix_nopp_file, allow_pickle=True)
        data_pp = np.load(matrix_pp_file, allow_pickle=True)

        #get the values of Vienna RNAFold
        bp_vrna = get_bp_counts(seqs, vrna)
        bp_count_truth.append(sum(bp_vrna.values()) / len(seqs))

        #check for unprocessed data
        bp_counts = 0   #bp counter
        ids = 0         #identity check counter (are there bp with itself)

        for a in data_no_pp:
            a = (a + a.T) / 2                   #make matrix symmetric
            id = a * np.identity(a.shape[0])    #check for 1s in diagonal (bp with itself)
            if torch.sum(id) != 0:
                ids += 1
            bp_counts += torch.sum(a)/2
        bp_count_no_pp.append(bp_counts/len(seqs))
        ids_nopp.append(ids)

        # check for postprocessed data
        ids = 0         #identity check counter (are there bp with itself)
        bp_counts = 0   #bp counter

        for a in data_pp:
            a = (a + a.T) / 2                   #make matrix symmetric
            bp_counts += torch.sum(a)/2
            id = a * np.identity(a.shape[0])    #check for 1s in diagonal (bp with itself)
            if torch.sum(id) != 0:
                ids += 1

        ids_pp.append(ids)
        bp_count_pp.append(bp_counts/len(seqs))     #get average # of bp pairs for length n postprocessed


    print(f"before postprocessing there are in {sum(ids_nopp)} self-loops in the {len(seqs)*len(n_lengths)} sequences")
    print(f"after postprocessing there are in {sum(ids_pp)} self-loops in  the {len(seqs)*len(n_lengths)} sequences")
    #plt.plot(n_lengths, bp_count_truth)
    # (x**2) * 0.002532 - x*0.021662 - 2.797929
    x = np.array(n_lengths)
    RNADeep = (x**2) * 0.002532 - x*0.021662 - 2.797929
    plt.plot(n_lengths, RNADeep)

    z = np.polyfit(x, bp_count_no_pp, 2)
    p = np.poly1d(z)
    plt.plot(x, p(x))

    z = np.polyfit(x, bp_count_pp, 1)
    p = np.poly1d(z)
    plt.plot(x, p(x))

    plt.ylim(0,250)

    # plt.plot(n_lengths, bp_count_no_pp)
    # plt.plot(n_lengths, bp_count_pp)

    plt.legend(["RNA Deep", "before postprocessing", "after postprocessing"])
    plt.xlabel("sequence length")
    plt.ylabel("number of bp")
    plt.title("Number of bp vs. length")
    plt.grid(color='black', linestyle='-', linewidth=0.1)
    if save:
        plt.savefig(save)
        print(f"Plot was saved as {save}.")
    plt.show()



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


def model_eval(contact_net, test_generator):
    '''
    function to evaluate the given UFold model without postprocessing and with postprocessing
    evaluates: MCC, F1, precision and recall
    arguments:
        contact_net: ..., UFold model, which should be tested
        test_generator: DataGenerator, generator with the dataset, which should be tested
    return:
        (mcc_no_pp, f1_no_pp, p_no_pp, r_no_pp), (mcc_pp, f1_pp, p_pp, r_pp)
            mcc_no_pp:  average MCC of the data set without postprocessing
            f1_no_pp:   average F1 of the data set without postprocessing
            p_no_pp:    average precision of the data set without postprocessing
            r_no_pp:    average recall of the data set without postprocessing
            mcc_pp:     average MCC of the data set with postprocessing
            f1_pp:      average F1 of the data set with postprocessing
            p_pp:       average precision of the data set with postprocessing
            r_pp:       average recall of the data set with postprocessing
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    contact_net.train()
    result_no_pp_train = list()
    result_pp_train = list()
    mcc_no_pp = 0
    mcc_pp = 0
    batch_n = 0
    run_time = []

    #iterate over our dataset
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


def analysis(file_path):
    '''
    function to analysis of real data and given prediction without postprocessing and with postprocessing
    evaluates:
        - structural features (External Loop (EL), Bulge Loop (BL), Hairpin Loop (HL), Internal Loop (IL), Multi Loop (ML))
        - Average number of structural element
        - relative frequencies of bp types
    arguments:
        file_path: str, path to the required files, files needed (file_path_sequence.npy, file_path_structure.npy)
            [npy files created by metrics.random_ml_forensic or metrics.fa2npy]
    return:
        None
        prints the results of analysis
    '''
    header = (f'Model paired exterior bulge hairpin interior multi '
              f'#helices #exterior #bulge #hairpin #interior #multi '
              f'base-pairs   %GC   %CG   %AU   %UA   %GU   %UG          %NC')
    #required files
    sequence_file = file_path + "_sequence.npy"
    structure_file = file_path + "_structure.npy"
    matrix_nopp_file = file_path + "_matrix_nopp.npy"
    matrix_pp_file = file_path + "_matrix_pp.npy"

    #load the sequences and structure of "Truth" (vienna RNA)
    seqs = np.load(sequence_file)
    vrna = np.load(structure_file)
    data_no_pp = np.load(matrix_nopp_file, allow_pickle=True)
    data_pp = np.load(matrix_pp_file, allow_pickle=True)

    l = max(len(s) for s in seqs)
    if min(len(s) for s in seqs) != l:
        l = 0

    # Show loop types from the MFE structures
    lt_vrna, lt_counts = julia_looptypes(seqs, vrna)
    bp_vrna = get_bp_counts(seqs, vrna)
    bp_tot_vrna = sum(bp_vrna.values())
    bp_vrna = {bp: cnt / bp_tot_vrna for bp, cnt in bp_vrna.items()}

    nc_vrna = sum([val for (i, j), val in bp_vrna.items() if not canon_bp(i, j)])
    # assert nc_vrna == 0
    print(header)
    print((f"{'vrna':5s}"
           f"{lt_vrna['S']:>7.3f} "
           f"{lt_vrna['E']:>8.3f} "
           f"{lt_vrna['B']:>5.3f} "
           f"{lt_vrna['H']:>7.3f} "
           f"{lt_vrna['I']:>8.3f} "
           f"{lt_vrna['M']:>5.3f} "
           f"{lt_counts['#S']:>8.3f} "
           f"{lt_counts['#E']:>9.3f} "
           f"{lt_counts['#B']:>6.3f} "
           f"{lt_counts['#H']:>8.3f} "
           f"{lt_counts['#I']:>9.3f} "
           f"{lt_counts['#M']:>6.3f} "
           f"{bp_tot_vrna:>10d} "
           f"{bp_vrna[('G', 'C')]:>5.3f} "
           f"{bp_vrna[('C', 'G')]:>5.3f} "
           f"{bp_vrna[('A', 'U')]:>5.3f} "
           f"{bp_vrna[('U', 'A')]:>5.3f} "
           f"{bp_vrna[('G', 'U')]:>5.3f} "
           f"{bp_vrna[('U', 'G')]:>5.3f} "
           f"{nc_vrna:>12.10f} "))

    nnss = julia_prediction(seqs, data_no_pp)
    # Show loop types from the neural network structures
    lt_nnss, lt_counts = julia_looptypes(seqs, nnss)
    bp_nnss = get_bp_counts(seqs, nnss)
    bp_tot_nnss = sum(bp_nnss.values())
    bp_nnss = {bp: cnt / bp_tot_nnss for bp, cnt in bp_nnss.items()}
    nc_nnss = sum([val for (i, j), val in bp_nnss.items() if not canon_bp(i, j)])
    print((f"{'no pp':5s}"
           f"{lt_nnss['S']:>7.3f} "
           f"{lt_nnss['E']:>8.3f} "
           f"{lt_nnss['B']:>5.3f} "
           f"{lt_nnss['H']:>7.3f} "
           f"{lt_nnss['I']:>8.3f} "
           f"{lt_nnss['M']:>5.3f} "
           f"{lt_counts['#S']:>8.3f} "
           f"{lt_counts['#E']:>9.3f} "
           f"{lt_counts['#B']:>6.3f} "
           f"{lt_counts['#H']:>8.3f} "
           f"{lt_counts['#I']:>9.3f} "
           f"{lt_counts['#M']:>6.3f} "
           f"{bp_tot_nnss:>10d} "
           f"{bp_nnss[('G', 'C')]:>5.3f} "
           f"{bp_nnss[('C', 'G')]:>5.3f} "
           f"{bp_nnss[('A', 'U')]:>5.3f} "
           f"{bp_nnss[('U', 'A')]:>5.3f} "
           f"{bp_nnss[('G', 'U')]:>5.3f} "
           f"{bp_nnss[('U', 'G')]:>5.3f} "
           f"{nc_nnss:>12.10f} "))

    nnss = julia_prediction(seqs, data_pp)
    # Show loop types from the neural network structures
    lt_nnss, lt_counts = julia_looptypes(seqs, nnss)
    bp_nnss = get_bp_counts(seqs, nnss)
    bp_tot_nnss = sum(bp_nnss.values())
    bp_nnss = {bp: cnt / bp_tot_nnss for bp, cnt in bp_nnss.items()}
    nc_nnss = sum([val for (i, j), val in bp_nnss.items() if not canon_bp(i, j)])
    print((f"{'pp':5s}"
           f"{lt_nnss['S']:>7.3f} "
           f"{lt_nnss['E']:>8.3f} "
           f"{lt_nnss['B']:>5.3f} "
           f"{lt_nnss['H']:>7.3f} "
           f"{lt_nnss['I']:>8.3f} "
           f"{lt_nnss['M']:>5.3f} "
           f"{lt_counts['#S']:>8.3f} "
           f"{lt_counts['#E']:>9.3f} "
           f"{lt_counts['#B']:>6.3f} "
           f"{lt_counts['#H']:>8.3f} "
           f"{lt_counts['#I']:>9.3f} "
           f"{lt_counts['#M']:>6.3f} "
           f"{bp_tot_nnss:>10d} "
           f"{bp_nnss[('G', 'C')]:>5.3f} "
           f"{bp_nnss[('C', 'G')]:>5.3f} "
           f"{bp_nnss[('A', 'U')]:>5.3f} "
           f"{bp_nnss[('U', 'A')]:>5.3f} "
           f"{bp_nnss[('G', 'U')]:>5.3f} "
           f"{bp_nnss[('U', 'G')]:>5.3f} "
           f"{nc_nnss:>12.10f} "))


    # In case you care, look at the test structures vs predicted structures!
    # for s, v, n in zip(seqs, vrna, nnss):
    #     print(f'# {s}')
    #     print(f'# {v} (test)')
    #     print(f'# {n} (pred)')


if __name__ == '__main__':
    folder = "data/analysis/length/length_test_ufold_model/"
    #folder = "data/analysis/type_analysis/N2000_n70_n100/"
    #length_analysis(folder)
    #prediction_file = "data/analysis/type_analysis/N2000_n70_n100/N2000_n100"
    #analysis(prediction_file)
    n_range = list(range(30,251,10))
    length_analysis(folder, n_range, "N1000")

