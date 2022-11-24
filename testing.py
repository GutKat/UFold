from ufold import utils

import architecture
from torch.optim import Adam
import torch
from torch import nn

model = architecture.UNET()

x = torch.rand(3,17,600,600)

#x = torch.reshape(x, (1,17,200,200))
outcome = model(x)
print(outcome)