import sys

import numpy as np
import pandas as pd

import torch
from model.DDoS_discriminator import Discriminator

import sentencepiece as spm

model = Discriminator(256)
state = torch.load(f='./ai/model/test.pt', map_location=torch.device('cpu'))
for key in list(state['model'].keys()):
    if key[-30:] == '._packed_params._packed_params':
        if 'fc' in key or 'generator' in key:
            state['model'][key[:-30] + '.bias'] = torch.dequantize(state['model'][key][1])
            state['model'][key[:-30] + '.weight'] = torch.dequantize(state['model'][key][0])
            state['model'].pop(key)
        else:
            state['model'][key[:-30]+'.weight'] = torch.dequantize(state['model'].pop(key)[0])
    elif 'generator.0' in key or 'emb' in key or 'weight' in key or 'bias' in key:
        state['model'][key] = torch.dequantize(state['model'][key])
    else:
        state['model'].pop(key)
model.load_state_dict(state['model'])

filename = sys.argv[1]
data = pd.read_json(filename)['_source']

sp = spm.SentencePieceProcessor()
vocab_file = "packet.model"
sp.load(vocab_file)


used_packet = ["_source:layers:frame:frame.time",
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

kind_of_attack = [["common_packet", 0],
                 ["TCPScan", 1],
                 ["UDPScan", 2],
                 ["NULLScan", 3]]


def get_value_from_stacked_dictionary(data, keys):
    if len(keys) == 1:
        return data[keys[0]]
    else:
        try:
            return get_value_from_stacked_dictionary(data[keys[0]], keys[1:])
        except Exception as e:
            return 'None'


for i, packet in enumerate(data.values):
    temp = []
    for packet_name in used_packet:
        packet_name = packet_name.split(":")
        temp.append(get_value_from_stacked_dictionary(packet, packet_name[1:]))

    temp = sp.encode_as_ids('<cls>'+'<sep>'.join(temp))
    data[i] = [temp, len(temp)]

max_length = 90
for d in data:
    gap = max_length - d[1]
    if gap > 0:
        d[0] = d[0] + [3] * gap  # [3] is pad token id
    else:
        d[0] = d[0][:max_length]
        d[1] = max_length

print(data)
data = [torch.tensor([i[0] for i in data]), [i[1] for i in data]]
y = model(data)
print(y)
for i in np.argmax(y.detach().numpy(), axis=1):
    print(i)

