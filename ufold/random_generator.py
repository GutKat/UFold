import RNA
import random
import numpy as np
import math
import data_generator


alphabet = ["A", "C", "G", "U"]
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
int_to_char = dict((i, c) for i, c in enumerate(alphabet))

def random_sequence(length):
    sequence = ''.join(random.choice('ACGU') for _ in range(length))
    return sequence


def generate_random_structures(lengths):
    sequences = []
    for length in lengths:
        seq = random_sequence(length)
        sequences.append((seq, *RNA.fold(seq)))
    return sequences


def one_hot_nuc_encode(nucleotide):
    one_hot = np.array([0,0,0,0])
    one_hot[char_to_int[nucleotide]] = 1
    return one_hot


def one_hot_encoder(sequence):
    one_hot = list(one_hot_nuc_encode(nuc) for nuc in sequence)
    one_hot_array = np.array((one_hot))
    return one_hot_array


def one_hot_nuc_decode(one_hot):
    return int_to_char[np.where(one_hot == 1)[0][0]]


def one_hot_decoder(array):
    sequence = list(one_hot_nuc_decode(num) for num in array)
    return "".join(sequence)


def generate_input(sequence):
    array = one_hot_encoder(sequence)
    n = len(sequence)
    input = np.zeros((17,n,n))
    k = 0
    while k < 16:
        for i in range(4):
            for j in range(4):
                kron_prod = np.kron(array[:,i], array[:,j].T)
                kron_prod = np.reshape(kron_prod, (1,n,n))
                input[k] = kron_prod
                k += 1
    input[-1] = data_generator.creatmat(sequence)
    return input
