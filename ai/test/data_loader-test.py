from ai.data_loader import PacketDataset
import sentencepiece as spm
from torch.utils.data import DataLoader

filepath = [["./ai/data/common_packet2.json", 0],
            ["./ai/data/TCPScan-2.json", 1],
            ["./ai/data/UDPScan-2.json", 2],
            ["./ai/data/NULLScan-2.json", 3]]

test_dataset = PacketDataset(filepath)
print(test_dataset[213])
print(len(test_dataset))


# sp = spm.SentencePieceProcessor()
# vocab_file = "packet.model"
# sp.load(vocab_file)
#
#
# temp = []
# for i in range(len(test_dataset)):
#     line = '<sep>'.join(test_dataset[i])
#     temp.append(len(sp.encode_as_ids(line)))
#
# print(temp)



def save_filtered_packet(filename='filtered_packet.txt'):
    with open(filename, 'a') as f:
        for i in range(len(test_dataset)):
            line = '<sep>'.join(test_dataset[i])
            f.write(line+'\n')


def collate(batch):
    return [[[i[0][0] for i in batch], [i[0][1] for i in batch]],
            [i[1] for i in batch]]


test_dataloader = DataLoader(test_dataset, batch_size=4, drop_last=True, collate_fn=collate, shuffle=True)

for i in test_dataloader:
    print(i)
    break
