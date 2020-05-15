import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, bidirectional=False):   # 4096 512  #
        super().__init__()
        self.hidden = hidden_size
        self.bidirection = bidirectional
        self.embedding = nn.Linear(input_size, hidden_size)      # word2vec
        self.bn = nn.BatchNorm1d(hidden_size, momentum=0.01)
        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)

    def forward(self, input):
        batch_size = input.size(0)
        input = input.view(batch_size*80, 4096)  # 128*80=
        embedded = self.bn(self.embedding(input).view(batch_size, 80, -1))  # bn((b,80,512))
        output = embedded       # GRU(x)    #  x     :  (batch, time_step, input_size)  (b,80,512)
        output, hidden = self.gru(output)   #  output:  (batch, time_step, output_size) (b,80, os)os=hid_size(512)*bi
                                            #  hidden:  (n_layers, batch,  hidden_size) (2, b,512)
        if self.bidirection:
            # Sum bidirectional outputs
            output = (output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:])
        return output, hidden


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, caption, lengths, hidden):
        output = self.embedding(caption)
        packed = pack_padded_sequence(output, lengths, batch_first=True) # 变长RNN
        output, hidden = self.gru(packed, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden, packed[1]


class VanillaDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, caption, hidden, encoder_outputs):
        batch_size = caption.size(0)
        output = self.embedding(caption)
        output = output.view(batch_size, 1, self.hidden_size)            # time_step=1
        output, hidden = self.gru(output, hidden)    # hidden=None
        output = output.squeeze(1)
        output = self.out(output)
        return output, hidden


class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super().__init__()
        self.method = method
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size*2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.size(0)
        max_len = encoder_outputs.size(1)

        attn_energies = Variable(torch.zeros(batch_size, max_len))  # B*S
        if torch.cuda.is_available():
            attn_energies = attn_energies.cuda()

        '''
        # For each batch of encoder outputs
        for b in range(batch_size):
        # Calculate energy for each encoder output
        for i in range(max_len):
        attn_energies[b, i] = self.score(hidden[b], encoder_outputs[b, i].unsqueeze(0))
        '''

        for b in range(batch_size):
            attn_energies[b] = self.score(hidden[b].unsqueeze(0), encoder_outputs[b])

        return F.softmax(attn_energies, dim=1).unsqueeze(1)

    def score(self, hidden, encoder_output):
        if self.method == 'dot':
            energy = torch.mm(hidden, encoder_output.transpose(0,1))
            return energy

        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = torch.mm(hidden, energy.transpose(0,1))
            return energy

        elif self.method == 'concat':
            hidden = hidden.repeat(encoder_output.size(0), 1)
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            energy = torch.mm(self.v, energy.transpose(0, 1))
            return energy



class BAttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, vocab_size,num_layers=1, dropout=0.1):   # hidden_size=512 vocab_size=v_size
        super(BAttnDecoderRNN, self).__init__()

        # Define parameters
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size            # how many words dic have
        self.num_layers = num_layers
        self.dropout = dropout

        # Define layers
        self.embedding = nn.Embedding(vocab_size, hidden_size)  # 定义nn.Embedding(2, 5)，这里的2表示有2个词，5表示5维度
        self.dropout = nn.Dropout(dropout)
        self.attn = Attn('concat', hidden_size)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, num_layers,dropout=dropout, batch_first=True)
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, word_input, last_hidden, encoder_outputs): # word_input:(b,len) last_hidden(2,b,512)
                                                                                    #encoder_outputs(b,80,512)
        # Get the embedding of the current input word (last output word)
        batch_size = word_input.size(0)
        word_embedded = self.embedding(word_input)
        word_embedded = self.dropout(word_embedded)
        word_embedded = word_embedded.view(batch_size, 1, -1)  # B x 1 x N

        # Calculate attention weights and apply to encoder outputs
        attn_weights = self.attn(last_hidden[-1], encoder_outputs)

        context = attn_weights.bmm(encoder_outputs)  # B x 1 x N      # batch mm
        # Combine embedded input word and attended context, run through RNN

        rnn_input = torch.cat((word_embedded, context), 2)
        rnn_input = F.relu(rnn_input)
        output, hidden = self.gru(rnn_input, last_hidden)
        # Final output layer
        output = output.squeeze(1)  # B x N
        output = self.out(output)

        # Return final output, hidden state
        return output, hidden
