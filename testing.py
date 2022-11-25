import random

from ufold import utils
from Network import U_Net
from torch.optim import Adam
import torch
from torch import nn
from ufold import random_generator


class random_input():
    def __init__(self, length):
        seq, ss, energy = random_generator.generate_random_seq_and_ss([length])[0]
        self.seq = seq
        self.ss = ss

random.seed(42)
N_seqs = 10
n_seq = 16*10
train_set = []
for i in range(N_seqs):
    random_seq = random_input(n_seq)
    train_set.append(random_seq)

model = U_Net(img_ch=17)
#model.train()
random_seq = random_generator.random_sequence(n_seq)
test1 = random_generator.generate_input(random_seq)
x = torch.rand(1, 17, n_seq, n_seq)
test1 = test1.reshape(1, 17, n_seq, n_seq)
outcome1 = model(test1)

print(outcome1)


