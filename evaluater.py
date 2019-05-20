import torch
import math
import logging
logger = logging.getLogger()


## ll for log-likelihood and coz we use LogSoftmax, the output is ll
def perplexity(model, dataloader, DEVICE):

    model.eval()  # test 时， 使 dropout BN 失效

    data_ll = []
    with torch.no_grad():
        for input_tensor, target_tensor, length_tensor in dataloader.tst_flow():
            input_tensor = input_tensor.to(DEVICE)
            output_tensor = model(input_tensor)  # (seq_len, 1, dic_size)
            output_tensor = output_tensor.squeeze(1)  # (seq_len, dic_size)

            sentence_ll = []
            for i in range(output_tensor.size(0)):
                target_index = target_tensor[i].item()
                token_ll = output_tensor[i, target_index].item()
                sentence_ll.append(token_ll)

            data_ll.append(sentence_ll)


        size = sum([len(sentence_ll) for sentence_ll in data_ll])
        sum_nll = sum([sum(sentence_ll) for sentence_ll in data_ll])

    avg_nll = sum_nll / size
    ppl = math.exp(-avg_nll)
    logger.info(ppl)
    print(ppl)

    model.train()  # 重新回归 train 模式



