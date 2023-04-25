import torch
import torch.nn as nn
import numpy as np
import pandas as pd

class Discriminator(nn.Module):
    def __init__(
            self,
            input_size,
            hidden_size,
            output_size,
            dropout_p=.1,
            max_length=512,
         ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

    def forward(self, x):
        pass