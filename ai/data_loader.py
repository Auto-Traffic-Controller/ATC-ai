import json
import tqdm

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

import sentencepiece as spm


def get_value_for_stacked_dictionary(data, keys):
    if len(keys) == 1:
        return data[keys[0]]
    else:
        try:
            return get_value_for_stacked_dictionary(data[keys[0]], keys[1:])
        except Exception as e:
            return 'None'


class PacketDataset(Dataset):

    def __init__(self, filepath, max_length=90):
        print("loading data...")
        self.data = pd.read_json(filepath)['_source']

        self.used_packet = ["_source:layers:frame:frame.time",
                            "_source:layers:frame:frame.number",
                            "_source:layers:eth:eth.src",
                            "_source:layers:eth:eth.dst",
                            "_source:layers:ip:ip.src_host",
                            "_source:layers:ip:ip.dst_host",
                            "_source:layers:tcp:tcp.flags_tree:tcp.flags.str",
                            "_source:layers:tcp:tcp.srcport",
                            "_source:layers:tcp:tcp.dstport",
                            "_source:layers:udp:udp.srcport",
                            "_source:layers:udp:udp.dstport",
                            "_source:layers:frame:frame.protocols",
                            "_source:layers:http:http.host"]
        self.sp = spm.SentencePieceProcessor()
        self.vocab_file = "packet.model"
        self.sp.load(self.vocab_file)

        # encoding
        for i, packet in enumerate(self.data.values):
            temp = []
            for packet_name in self.used_packet:
                packet_name = packet_name.split(":")
                temp.append(get_value_for_stacked_dictionary(packet, packet_name[1:]))

            temp = self.sp.encode_as_ids('<cls>'+'<sep>'.join(temp))
            self.data[i] = [temp, len(temp)]

        # padding
        for data in self.data:
            gap = max_length - data[1]
            if gap > 0:
                data[0] = data[0] + [3] * gap   # [3] is pad token id
            else:
                continue

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

