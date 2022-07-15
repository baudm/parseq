import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):

    def __init__(self, input_size, hidden_size, num_class, num_char_embeddings=256):
        super().__init__()
        self.attention_cell = AttentionCell(input_size, hidden_size, num_char_embeddings)
        self.hidden_size = hidden_size
        self.num_class = num_class
        self.generator = nn.Linear(hidden_size, num_class)
        self.char_embeddings = nn.Embedding(num_class, num_char_embeddings)

    def forward(self, batch_H, text, max_label_length=25):
        """
        input:
            batch_H : contextual_feature H = hidden state of encoder. [batch_size x num_steps x num_class]
            text : the text-index of each image. [batch_size x (max_length+1)]. +1 for [SOS] token. text[:, 0] = [SOS].
        output: probability distribution at each step [batch_size x num_steps x num_class]
        """
        batch_size = batch_H.size(0)
        num_steps = max_label_length + 1  # +1 for [EOS] at end of sentence.

        output_hiddens = batch_H.new_zeros((batch_size, num_steps, self.hidden_size), dtype=torch.float)
        hidden = (batch_H.new_zeros((batch_size, self.hidden_size), dtype=torch.float),
                  batch_H.new_zeros((batch_size, self.hidden_size), dtype=torch.float))

        if self.training:
            for i in range(num_steps):
                char_embeddings = self.char_embeddings(text[:, i])
                # hidden : decoder's hidden s_{t-1}, batch_H : encoder's hidden H, char_embeddings : f(y_{t-1})
                hidden, alpha = self.attention_cell(hidden, batch_H, char_embeddings)
                output_hiddens[:, i, :] = hidden[0]  # LSTM hidden index (0: hidden, 1: Cell)
            probs = self.generator(output_hiddens)

        else:
            targets = text[0].expand(batch_size)  # should be fill with [SOS] token
            probs = batch_H.new_zeros((batch_size, num_steps, self.num_class), dtype=torch.float)

            for i in range(num_steps):
                char_embeddings = self.char_embeddings(targets)
                hidden, alpha = self.attention_cell(hidden, batch_H, char_embeddings)
                probs_step = self.generator(hidden[0])
                probs[:, i, :] = probs_step
                _, next_input = probs_step.max(1)
                targets = next_input

        return probs  # batch_size x num_steps x num_class


class AttentionCell(nn.Module):

    def __init__(self, input_size, hidden_size, num_embeddings):
        super().__init__()
        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)  # either i2i or h2h should have bias
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.rnn = nn.LSTMCell(input_size + num_embeddings, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, prev_hidden, batch_H, char_embeddings):
        # [batch_size x num_encoder_step x num_channel] -> [batch_size x num_encoder_step x hidden_size]
        batch_H_proj = self.i2h(batch_H)
        prev_hidden_proj = self.h2h(prev_hidden[0]).unsqueeze(1)
        e = self.score(torch.tanh(batch_H_proj + prev_hidden_proj))  # batch_size x num_encoder_step * 1

        alpha = F.softmax(e, dim=1)
        context = torch.bmm(alpha.permute(0, 2, 1), batch_H).squeeze(1)  # batch_size x num_channel
        concat_context = torch.cat([context, char_embeddings], 1)  # batch_size x (num_channel + num_embedding)
        cur_hidden = self.rnn(concat_context, prev_hidden)
        return cur_hidden, alpha
