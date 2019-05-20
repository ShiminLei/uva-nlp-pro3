import torch
from torch import nn
import time
from evaluater import perplexity


def train(model, dataloader, config):
    DEVICE = torch.device("cuda" if config.use_gpu else "cpu")
    print('device:', DEVICE)

    model.to(DEVICE)
    loss = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    start_iter = 0

    total_loss = 0  # for print

    for i in range(start_iter + 1, config.iter_num + 1):

        input_tensor, target_tensor, length_tensor = dataloader.random_batch()

        input_tensor = input_tensor.to(DEVICE)
        target_tensor = target_tensor.to(DEVICE)
        length_tensor = length_tensor.to(DEVICE)
        # forward
        output = model(input_tensor)
        output_tensor = nn.utils.rnn.pack_padded_sequence(output, length_tensor).data  # shape (seq_length) sum of each sentence length

        # 计算 目标函数
        output_loss = loss(output_tensor, target_tensor)
        output_loss.backward()  # 计算梯度
        nn.utils.clip_grad_norm_(model.parameters(), 5) # 对梯度剪裁
        optimizer.step()  # 梯度下降
        optimizer.zero_grad() # 梯度归零

        total_loss += output_loss.item()   # item 取出数字

        if i % config.print_every == 0:
            print('%s iter(%d) avg-loss: %.4f' % (time.strftime('%x %X'), i, total_loss / config.print_every))
            total_loss = 0

            perplexity(model, dataloader, DEVICE)



    print('training ends')
