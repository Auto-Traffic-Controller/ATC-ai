from ai.data_loader import PacketDataset

filepath = "ai/data/train-test-tcp-udp-icmp.json"

test_dataset = PacketDataset(filepath)
print(test_dataset.data)
