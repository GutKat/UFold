import random
import numpy as np
from ufold import utils
from ufold import random_generator, postprocess, metrics
import os
from tqdm import tqdm


class random_input():
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


def create_bbseq_file(sample, path):
    filename = os.path.split(path)[1]
    pairs = ct2struct(sample.ss)
    paired = [0] * len(sample.ss)
    for pair in pairs:
        le = pair[0]
        ri = pair[1]
        paired[le] = ri + 1
        paired[ri] = le + 1
    bbseq = {"seq": sample.seq, "pair": paired}
    with open(path, "w") as f:
        seq = bbseq["seq"]
        pairs = bbseq["pair"]
        n = len(seq)
        for index in range(n):
            line = [str(index+1), str(seq[index]), str(pairs[index])]
            line = " ".join(line)
            f.write(line)
            if index != n-1:
                f.write("\n")
    return None


def bp_file(N_seqs, n_seq, purpose="train"):
    seed_dict = {"val": 10, "test": 20, "train": 30}
    seed = seed_dict[purpose]

    utils.seed_torch(seed)
    random.seed(seed)

    folder_path = f"data/random/raw/N{N_seqs}_n{n_seq}_{purpose}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    files = []
    for i in tqdm(range(N_seqs)):
        test = random_input(n_seq)
        files.append(test)
        create_bbseq_file(test, f"{folder_path}/test{i}.txt")
    print(f"finish creating {N_seqs} random sequences with length of {n_seq}")
#

def create_fa(N_seqs, n_seq, seed=42):
    utils.seed_torch(seed)
    random.seed(seed)
    file_fa = f"data/input.txt"
    file_fa_ss = f"data/random/raw/N{N_seqs}_n{n_seq}_ss.fa"
    data = []
    for i in range(N_seqs):
        data.append(random_input(n_seq))

    with open(file_fa, "w") as f:
        for i, dat in enumerate(data):
            f.write(f">random_{i+1}\n")
            f.write(dat.seq)
            f.write("\n")
    with open(file_fa_ss, "w") as f:
        for i, dat in enumerate(data):
            f.write(f">random_{i+1}\n")
            f.write(f"{dat.seq}\n{dat.ss}")
            f.write("\n")

if __name__ == '__main__':
    #pass
    N_seqs = 50000
    n_seq = 100
    purpose = "train"
    bp_file(N_seqs=N_seqs, n_seq=n_seq, purpose=purpose)
    #create_fa(N_seqs=N_seqs, n_seq=n_seq)
    # true = "data/random/N100_n160_ss.fa"
    # pred = "results/input_dot_ct_file.txt"
    # result = metrics.prediction_evaluation(pred, true)


