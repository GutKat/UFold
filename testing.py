import random

from ufold import utils

from Network import U_Net
from torch.optim import Adam
import torch
from torch import nn
from ufold import random_generator

random.seed(42)
N_seqs = 10
n_seq = 160
train_set = []
for i in range(N_seqs):
    seq = random_generator.random_sequence(n_seq)
    train_set.append(random_generator.generate_input(seq))

model = U_Net(img_ch=17)


#outcome1 = model(x)

