
from torch import nn

class LM(nn.Module):
    ''' Language Model '''

    def __init__(self, dic_size, input_size, hidden_size, layer_size):
        super(LM, self).__init__()

        self.embedding = nn.Embedding(dic_size, input_size)
        self.embedding_dropout = nn.Dropout(p=0.2)
        self.lstm = nn.LSTM(input_size, hidden_size, layer_size)
        self.linear = nn.Linear(hidden_size, dic_size) # 把hidden size 变成 dic size
        self.log_softmax = nn.LogSoftmax(dim=2)


    def forward(self, input_tensor):          # input_tensor   shape: (seq_legnth, batch) | shape(batch)
        embedded = self.embedding(input_tensor)  # shape:(seq_length, batch, input_size)
        embedded = self.embedding_dropout(embedded) # shape:(seq_length, batch, input_size)

        lstm_output, _ = self.lstm(embedded) # shape: (seq_length, batch, hidden_size)

        output = self.linear(lstm_output)       # shape: (seq_length, batch, dic_size)
        output = self.log_softmax(output)

        return output