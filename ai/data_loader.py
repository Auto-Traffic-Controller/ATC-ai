import tqdm
import pandas as pd
from torch.utils.data import Dataset

import sentencepiece as spm


def get_value_from_stacked_dictionary(data, keys):
    if len(keys) == 1:
        return data[keys[0]]
    else:
        try:
            return get_value_from_stacked_dictionary(data[keys[0]], keys[1:])
        except Exception as e:
            return 'None'


class PacketDataset(Dataset):

    def __init__(self, filepath, max_length=90):

        print("loading data...")
        self.data = pd.DataFrame()
        for fp in filepath:
            data = pd.read_json(fp[0])
            y = fp[1]
            data['y'] = y
            print(data)
            if self.data.empty:
                self.data = data
            else:
                self.data = pd.concat([self.data, data], ignore_index=True)
        self.tgt = self.data['y']
        self.data = self.data['_source']
        print("all data loaded")

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

        print("data encoding...")
        # encoding
        for i, packet in tqdm.tqdm(enumerate(self.data.values)):
            temp = []
            for packet_name in self.used_packet:
                packet_name = packet_name.split(":")
                temp.append(get_value_from_stacked_dictionary(packet, packet_name[1:]))

            temp = self.sp.encode_as_ids('<cls>'+'<sep>'.join(temp))
            self.data[i] = [temp, len(temp)]
        print("all data encoded")

        print("data padding")
        # padding
        for data in tqdm.tqdm(self.data):
            gap = max_length - data[1]
            if gap > 0:
                data[0] = data[0] + [3] * gap   # [3] is pad token id
            else:
                continue
        print("all data padded")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.tgt[idx]

