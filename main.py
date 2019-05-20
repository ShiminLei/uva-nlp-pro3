import os
import argparse
import logging

from dataloader import DataLoader
from model import LM
from engine import train
from utils import log_setp

logger = logging.getLogger()

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='minibatch_rnnlm')
parser.add_argument('--cur_dir', type=str, default=os.path.dirname(__file__))
parser.add_argument('--log_dir', type=str, default='./log')

parser.add_argument('--trn_file', type=str, default='trn-wiki.txt')
parser.add_argument('--tst_file', type=str, default='tst-wiki.txt')

parser.add_argument('--input_size', type=int, default=32)
parser.add_argument('--hidden_size', type=int, default=64)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--layer_num', type=int, default=1)

parser.add_argument('--print_every', type=int, default=5)
parser.add_argument('--iter_num', type=int, default=5000)
parser.add_argument('--use_gpu', type=bool, default=False)

args = parser.parse_args()


def main(config):
    log_setp(logger, config)

    dataloader = DataLoader(config)
    word_map = dataloader.word_map
    print('number of tokens:', len(word_map))

    print(f'create model: input size {config.input_size}, hidden size {config.hidden_size}, layer number {config.layer_num}')
    model = LM(len(word_map), config.input_size, config.hidden_size, config.layer_num)

    print(f'start training of {config.iter_num} iterations')
    train(model, dataloader, config)



if __name__ == '__main__':
    config, unparsed = parser.parse_known_args()
    main(config)
