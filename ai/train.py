import argparse

from ai.model.DDoS_discriminator import Discriminator
from ai.data_loader import PacketDataset

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--filepath',
                   type=str,
                   default='./data',
                   help='path of train data')
    p.add_argument(
                   '--n_epochs',
                   type=int,
                   default=4,
                   help='Number of epochs to train. Default=%(default)s')
    p.add_argument('--batch_size',
                   type=int,
                   default=64,
                   help='Number of data in a iteration')
    p.add_argument('--hidden_size',
                   type=int,
                   default=256,
                   help='dimension size of data')
    p.add_argument('--lr',
                   type=float,
                   default=0.9,
                   help='learning rate')

    return p.parse_args(args=[])


config = define_argparser()

filepath = "./ai/data/X-MASScan-2.json"
packet_dataset = PacketDataset(config.filepath)


def collate(batch):
    return [[[i[0][0] for i in batch], [i[0][1] for i in batch]],
            [i[1] for i in batch]]

device = torch.device('cuda:0')
packet_dataloader = DataLoader(packet_dataset, batch_size=config.batch_size, drop_last=True, collate_fn=collate)
model = Discriminator(config.hidden_size).to(device)
crit = nn.NLLLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, betas=(.9, .98))


for epoch in range(config.n_epochs):
    for mini_batch in packet_dataloader:
        y = mini_batch[1]
        y_hat = model(mini_batch[0].to(device))

        loss = crit(y_hat, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

model_fn = './model/test'
torch.save(
    {
        'model': model.state_dict(),
        'opt': optimizer.state_dict(),
        'config': config,
    }, model_fn
)





