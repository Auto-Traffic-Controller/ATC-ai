import sentencepiece as spm

spm.SentencePieceTrainer.Train(f'--input="filtered_packet.txt" '
                               '--model_prefix=packet '
                               '--vocab_size=600 '
                               '--model_type=bpe '
                               '--max_sentence_length=500000 '                               
                               '--user_defined_symbols=<pad>,None,<sep>,<cls>')

