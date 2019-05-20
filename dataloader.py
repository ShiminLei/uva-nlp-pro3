import os
import torch
from torch import nn
import random

class DataLoader:
    def __init__(self,config):
        self.config = config
        # train sets
        self.lines = self.load_data(self.config.trn_file)
        self.word_map = self.build_word_map(self.lines)
        self.lines_idx = self.data_to_idx(self.lines)
        # test sets
        self.tst_lines = self.load_data(self.config.tst_file)
        self.tst_lines_idx = self.data_to_idx(self.tst_lines)


    def load_data(self, file_path):
        path = os.path.join(self.config.cur_dir, file_path)
        with open(path, 'r', encoding='utf-8') as file:
            lines = [l.strip() for l in file.read().split('\n')]
        lines = [l.split(' ') for l in lines if l != '']
        token_num = sum([len(l) for l in lines])
        print(f'{file_path}     #sentences {len(lines)}, #tokens {token_num}')
        print(lines[0])
        return lines

    def build_word_map(self, lines):
        tokens = set()
        for line in lines:
            for token in line:
                tokens.add(token)
        tokens = sorted(tokens)  # sort the set in order to stable the word index
        return {token: idx for idx, token in enumerate(tokens)}

    def data_to_idx(self, lines):
        return [[self.word_map[token] for token in line] for line in lines]

    def random_batch(self):
        sentences = [random.choice(self.lines_idx) for _ in range(self.config.batch_size)]
        sentences.sort(key=len, reverse=True)
        lengths = [len(s) - 1 for s in sentences]  # # len includes the <start> <stop>, so need -1

        input_tensors_seq = [self.numpy_to_tensors(s[:-1]) for s in sentences]
        target_tensors_seq = [self.numpy_to_tensors(s[1:]) for s in sentences]  # 16个长短不一致的列表
        # print(len(target_tensors_seq[0]))
        # print(len(target_tensors_seq[1]))

        # any padding value is fine, coz it will packed with actual length in the model
        input_tensor = nn.utils.rnn.pad_sequence(input_tensors_seq)  # shape:(最长长度, batch-size)
        # print(input_tensor.shape)

        # shape (token_length) sum of each sentence length
        target_tensor = nn.utils.rnn.pack_sequence(target_tensors_seq).data
        # print(target_tensor.shape)

        return input_tensor, target_tensor, torch.tensor(lengths)

    def numpy_to_tensors(self, sentence):
        tensors = torch.LongTensor(sentence)
        return tensors

    def tst_flow(self):
        for sentence in self.tst_lines_idx:
            sentences = [sentence]
            lengths = [len(s) - 1 for s in sentences]  # # len includes the <start> <stop>, so need -1

            input_tensors_seq = [self.numpy_to_tensors(s[:-1]) for s in sentences]
            target_tensors_seq = [self.numpy_to_tensors(s[1:]) for s in sentences]  # 16个长短不一致的列表

            input_tensor = nn.utils.rnn.pad_sequence(input_tensors_seq)  # shape:(最长长度, batch-size)
            target_tensor = nn.utils.rnn.pack_sequence(target_tensors_seq).data

            yield input_tensor, target_tensor, torch.tensor(lengths)



