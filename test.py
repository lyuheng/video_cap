import numpy as np
# a = [[1,2],[3,4]]
# a = np.array(a)
# print(a)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, bidirectional=False):   # 4096 512  #
        super().__init__()
        self.hidden_size = hidden_size
        self.bidirection = bidirectional
        self.embedding = nn.Linear(input_size, hidden_size)
        #self.bn = nn.BatchNorm1d(hidden_size)
        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)

    def forward(self, input):
        batch_size = input.size(0)
        input = input.view(batch_size*80, 4096)  # 128*80=
        embedded = self.embedding(input).view(batch_size, 80, -1)  # bn((b,80,512))
        output = embedded       # GRU(x)    #  x     :  (batch, time_step, input_size)  (b,80,512)
        output, hidden = self.gru(output)   #  output:  (batch, time_step, output_size) (b,80, os)os=512*bi
                                            #  hidden:  (n_layers, batch,  hidden_size) (2, b,512)
        if self.bidirection:
            # Sum bidirectional outputs
            output = (output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:])
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

    def forward(self, hidden, encoder_outputs):  # (16,512)  (16,80,512)
        batch_size = encoder_outputs.size(0)
        max_len = encoder_outputs.size(1)

        attn_energies = Variable(torch.zeros(batch_size, max_len))  # B*S (16,80)

        '''
        # For each batch of encoder outputs
        for b in range(batch_size):
        # Calculate energy for each encoder output
        for i in range(max_len):
        attn_energies[b, i] = self.score(hidden[b], encoder_outputs[b, i].unsqueeze(0))
        '''

        for b in range(batch_size):        # cal alpha      # match module
            attn_energies[b] = self.score(hidden[b].unsqueeze(0), encoder_outputs[b])    #(1,512) (80,512)

        #print('attn_energies', attn_energies.shape) (16,80)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)  # do softmax to alpha  return alpha

    def score(self, hidden, encoder_output):  #(1,512) (80,512)
        if self.method == 'dot':
            energy = torch.mm(hidden, encoder_output.transpose(0,1))    # (0,1) 和 (1,0) 都行
            return energy

        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = torch.mm(hidden, energy.transpose(0,1))
            return energy

        elif self.method == 'concat':
            hidden = hidden.repeat(encoder_output.size(0), 1)  #80,512
            energy = self.attn(torch.cat((hidden, encoder_output), 1))  # 80,1024 -> 80,512
            energy = torch.mm(self.v, energy.transpose(0, 1))   # 1,512 * 512, 80 = 1*80
            return energy



class BAttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, vocab_size,num_layers=2, dropout=0.1):   # hidden_size=512 vocab_size=v_size
        super(BAttnDecoderRNN, self).__init__()

        # Define parameters
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
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
        batch_size = word_input.size(0)             # [16, 1]
        word_embedded = self.embedding(word_input)  # [16, 512]
        print('\rword_embedded',word_embedded.shape)
        word_embedded = self.dropout(word_embedded) # [16, 512]
        print('\rword_embedded',word_embedded.shape)
        word_embedded = word_embedded.view(batch_size, 1, -1)  # B x 1 x N  [16, 1, 512]
        print('\rword_embedded',word_embedded.shape)

        # Calculate attention weights and apply to encoder outputs
        attn_weights = self.attn(last_hidden[-1], encoder_outputs) # [16, 1, 80]
        print('\rattn_weights', attn_weights.shape)
        context = attn_weights.bmm(encoder_outputs)  # B x 1 x N      # batch mm  [16, 1, 512]
        # Combine embedded input word and attended context, run through RNN

        print('\rcontext', context.shape)
        rnn_input = torch.cat((word_embedded, context), 2)   # [16, 1, 1024]
        print('\rrnn_input', rnn_input.shape)
        rnn_input = F.relu(rnn_input)                        # [16, 1, 1024]
        print('\rrnn_input', rnn_input.shape)
        output, hidden = self.gru(rnn_input, last_hidden)
        # Final output layer.   # [16, 1, 512]    [2, 16, 512]
        print('\r', output.shape, hidden.shape)
        output = output.squeeze(1)  # B x N
        print('\routput', output.shape)     # [16, 512]
        output = self.out(output)
        print('\routput', output.shape)     # [16, 50]

        # Return final output, hidden state
        return output, hidden               # [16, 50] [2, 16, 512]


#batch_size = 16
encoder = EncoderRNN(4096, 512)
image = torch.FloatTensor(np.ones((16,80,4096)))
encoder_outputs, encoder_hidden = encoder(image)  # [16, 80, 512]  [2, 16, 512]
#print(encoder_output.shape, encoder_hidden.shape)
# vocal_size = 50
decoder = BAttnDecoderRNN(512,50)
# # batch_size = 16 cap_len = 16
word_input = torch.LongTensor(np.ones((16)))
decoder_hidden = encoder_hidden[:2]

a,b = decoder(word_input, decoder_hidden, encoder_outputs)
print('a.shape: ', a.shape, 'b.shape: ', b.shape)

