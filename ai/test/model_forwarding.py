from ai.model.DDoS_discriminator import Discriminator
from ai.data_loader import PacketDataset
from torch.utils.data import DataLoader

import torch

filepath = "./ai/data/X-MASScan-2.json"
test_dataset = PacketDataset(filepath)

def collate(batch):
    return [torch.tensor([i[0] for i in batch]), torch.tensor([i[1] for i in batch])]


test_dataloader = DataLoader(test_dataset, batch_size=4, drop_last=True, collate_fn=collate)
model = Discriminator(256)
for i in test_dataloader:
    y = model(i)
    print(y)
    break
