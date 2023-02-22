'''
Author: Katrin Gutenbrunner
These functions have been created and used in the purpose of the Softwareproject 2022/23
In here are functions, which convert files into a specific file format (bbseq), which is used by UFold for
training, testing and predicting or create random data and store it in specific files (bbseq for UFold or nseq for ml_fornsic)
'''
import random
import numpy as np
from ufold import utils
from ufold import random_generator
import os
from tqdm import tqdm
import collections
import re
import pandas as pd
from pathlib import Path


class random_sample():
    '''
    class for creating random sequences with the corrseponding secondary structure
    secondary structure is predicted by using Vienna RNA
    arguments:
        length: int, defines the length of the sequence
    secondary structure = random_sample.ss
    sequence = random_sample.seq
    '''
    def __init__(self, length):
        seq, ss, energy = random_generator.generate_random_seq_and_ss([length])[0]
        self.seq = seq
        self.ss = ss


def ct2struct(ct):
    stack = list()
    struct = list()
    for i in range(len(ct)):
        if ct[i] == '(':
            stack.append(i)
        if ct[i] == ')':
            left = stack.pop()
            struct.append([left, i])
    return struct


def sample2bbseq(seq, ss, path):
    '''
    function to create a bb sequence file from one given sample (sequence and secondary structure)
    bbseq files are needed for creating (c)Pickles files (which are used by Ufold)
    arguments:
        seq: str, sequence of RNA
        ss: ss, secondary structure of RNA
        path: str, path where the bbseq file should be saved
    return:
        none, only creates bbseq file
    '''
    #create the pairs of the secondary structure
    pairs = ct2struct(ss)
    paired = [0] * len(ss)
    #create the pair for the bbseq file
    for pair in pairs:
        le = pair[0]
        ri = pair[1]
        paired[le] = ri + 1
        paired[ri] = le + 1
    with open(path, "w") as f:
        n = len(seq)
        for index in range(n):
            line = [str(index+1), str(seq[index]), str(paired[index])]
            line = " ".join(line)
            f.write(line)
            if index != n-1:
                f.write("\n")
    return None


def random_bbseq(N_seqs, n_seq, purpose="train", seed_set=False, folder_path=None):
    '''
    function to create a folder with bpseq files of N_seqs random sequences of length n_seq
    arguments:
        N_seqs: int, number of sequences
        n_seq: int, length of each sequence
        purpose: str, sets specific seeds for reproducibility, accepty "val" (validation), "test" (testing), "train" (training)
        seed_set: int, instead of purpose also a specific seed can be set, used for reproducibility
        folder_path: str, path to the folder in which the bbseq files should be stored, if no folder_path is given, a folder is create with the given purpose or seed
    return:
        None
        creates the bbseq files within the folder_path
    '''
    #set seed according to the parameter, either purpose or seed_set
    seed_dict = {"val": 10, "test": 20, "train": 30}
    seed = seed_dict[purpose]
    if seed_set:
        seed = seed_set
    utils.seed_torch(seed)
    random.seed(seed)

    # check if specific folder is given if not, create folder according to purpose or seed
    if folder_path:
        # check if folder already exist, if not create it
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
    else:
        if seed_set:
            folder_path = f"data/random/raw/N{N_seqs}_n{n_seq}_{seed}"
        else:
            folder_path = f"data/random/raw/N{N_seqs}_n{n_seq}_{purpose}"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    # create the N_seqs sequences
    for _ in tqdm(range(N_seqs)):
        #create the sample with length n_seq
        sample = random_sample(n_seq)
        sample2bbseq(sample.seq, sample.ss, f"{folder_path}/len{n_seq}_{_+1}.txt")
    print(f"finish creating {folder_path}")


def random_ml_forensic(N_seqs, n_seq, output_folder, seed=42):
    '''
    function to create files for ml_forensic.py
    creates N_seqs random sequences of length n_seq,
    from these sequences the following is created within the output_folder:
        - txt files with sequences
        - npy file with sequences
        - npy file with secondary structures
    arguments:
        N_seqs: int, number of sequences
        n_seq: int, length of each sequence
        seed: int, sets seed for reproducibility
        output_folder: str, path to the folder in which the files should be stored
    return:
        None
        creates the txt file with name "N_seqs_n_seq.txt"
        creates the npy file with name "N_seqs_n_seq_sequence.npy"
        creates the npy file with name "N_seqs_n_seq_structure.npy"
    '''
    #set the given seed
    utils.seed_torch(seed)
    random.seed(seed)

    #create the sampels
    data = []
    for _ in tqdm(range(N_seqs)):
        data.append(random_sample(n_seq))

    #create the filename within the output folder
    seq_txt_file = f"{output_folder}/N{N_seqs}_n{n_seq}.txt"
    sequence_file = output_folder + f"/N{N_seqs}_n{n_seq}_sequence.npy"
    structure_file = output_folder + f"/N{N_seqs}_n{n_seq}_structure.npy"

    seqs = []
    structures = []
    #write all sequences of the samples within txt_file
    with open(seq_txt_file, "w") as f:
        for i, dat in tqdm(enumerate(data)):
            f.write(f">random_{i+1}\n")
            f.write(f"{dat.seq}\n")#{dat.ss}
            f.write("\n")
            seqs.append(dat.seq)
            structures.append(dat.ss)

    #store sequences and structures within the sequence_file and structure file
    np.save(sequence_file, seqs)
    np.save(structure_file, structures)


def fa2npy(fa_file, output_folder):
    '''
    function to create files for ml_forensic.py of a existing fa file. Creates:
        - txt files with sequences
        - npy file with sequences
        - npy file with secondary structures
    arguments:
        fa_file: str, path to the fa file, which should be used
        output_folder: str, path to the folder in which the files should be stored
    return:
        None
        creates the txt file with name "fa_file.txt"
        creates the npy file with name "fa_file_sequence.npy"
        creates the npy file with name "fa_file_structure.npy"
    '''
    #load the fa file
    with open(fa_file, "r") as f:
        data = f.readlines()

    #pattern to get name of sequences
    pattern = ">(.*) en"

    #create names for new files
    fa_file_stem = Path(fa_file).stem
    sequence_file = output_folder + f"{fa_file_stem}_sequence.npy"
    structure_file = output_folder + f"{fa_file_stem}_structure.npy"
    new_file = output_folder + f"{fa_file_stem}.txt"

    #storing sequence and structures
    seqs = []
    structures = []

    #if folder path is not existing, create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    #write into the new file
    with open(new_file, "w") as f:
        for i in tqdm(range(0, len(data), 3)):
            name = re.search(pattern, data[i])[1]
            seq = data[i + 1]
            seq = seq.replace("\n", "")
            ss = data[i + 2]
            ss = ss.replace("\n", "")
            seqs.append(seq)
            structures.append(ss)
            f.write(f">{name}\n")
            f.write(f"{seq}\n")  # {dat.ss}
            f.write("\n")
    #store sequences and structures in npy files
    np.save(sequence_file, seqs)
    np.save(structure_file, structures)


def pickle2fa(pickle_file, fa_file):
    '''
    function to convert a pickle file to a fa file
    arguments:
        pickle_file: str, path to the pickle file, from which the data should be collected
        fa_file: str, path to the fa file, in which the data should be stored
    return:
        None
        creates the fa_file.fa
    '''
    RNA_SS_data = collections.namedtuple('RNA_SS_data', 'seq ss_label length name pairs')
    #load data from pickle_file
    data = pd.read_pickle(pickle_file)
    names = []
    sequences = []
    structures = []
    for obj in tqdm(data):
        seq = obj[0]
        length = obj[2]
        name = obj[3]
        pairs = obj[4]
        #convert seq_encoding to sequences
        seq = utils.encoding2seq(seq)[0:length]
        #only use sequences with defined nucleotides (A,C,G or U)
        if "." in seq:
            continue

        #create the secodnary structure from the pair list
        ss = np.array(list(length * "."))
        for pair in pairs:
            if pair[0] < pair[1]:
                ss[pair[0]] = "("
                ss[pair[1]] = ")"
            else:
                ss[pair[1]] = "("
                ss[pair[0]] = ")"
        ss = "".join(ss)

        #save name sequences and structures
        names.append(name)
        sequences.append(seq)
        structures.append(ss)

    #create the fa file
    with open(fa_file, "w") as f:
        for i in range(len(names)):
            if i != len(names):
                f.write(f">{names[i]}\n")
                f.write(f"{sequences[i]}\n")
                f.write(f"{structures[i]}\n")
            # if we get to the last entry, we do not want to write a new line (\n)
            else:
                f.write(f">{names[i]}\n")
                f.write(f"{sequences[i]}\n")
                f.write(f"{structures[i]}")


if __name__ == '__main__':
    RNA_SS_data = collections.namedtuple('RNA_SS_data', 'seq ss_label length name pairs')
    n = 100
    purpose = "train"
    for N in [50000, 75000, 100000, 125000]:
        random_bbseq(N,n,purpose=purpose,folder_path=f"data/random/length_test/n{n}/N{N}_n{n}_{purpose}")
    # # output_file = "data/analysis/length/test"
    # # for n in range(30,251,10):
    # #     output_folder = f"data/analysis/length/length_test"
    # #     create_fa(1000, n, output_folder)
    # # filepath = "data/original/TS0.cPickle"
    # # fa_file = "data/TS0.fa"
    # # pickle_to_fa(filepath, fa_file)
    # # for n in range(30, 251, 10):
    # #     bp_file(1000, n, seed_set=20, folder_path=f"data/analysis/length/length_test_{n}")
    # output_folder = "/Users/katringutenbrunner/Desktop/UFold/data/rnadeep/N2000_n70/"
    # create_fa(2000, 70, output_folder)
    # #filename = "data/rnadeep/inv_120_10000.fa.txt"
    # #create_files_from_fa(filename, new_file)
    # # N_seq = 5000
    # # n_seq = 100
    # # #seeds = [1, 2, 3]
    # # purpose = "train"


