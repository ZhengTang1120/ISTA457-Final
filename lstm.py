import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class LSTM(nn.Module):

    def __init__(self, wembedding_dim, cembedding_dim, chidden_dim, hidden_dim, vocab_size, chars_size, target_size):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.chidden_dim = chidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, wembedding_dim)
        self.char_embeddings = nn.Embedding(chars_size, cembedding_dim)

        self.clstm = nn.LSTM(cembedding_dim, chidden_dim)
        self.lstm = nn.LSTMCell(wembedding_dim+chidden_dim, hidden_dim)

        self.hidden2tag = nn.Linear(hidden_dim, target_size)

        self.hidden = self.init_hidden()
        self.chidden = self.init_chidden()


    def init_hidden(self):
        return (autograd.Variable(torch.zeros(1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, self.hidden_dim)))

    def init_chidden(self):
        return (autograd.Variable(torch.zeros(1, 1, self.chidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.chidden_dim)))

    def forward(self, tweet):
        wembeds = self.word_embeddings(autograd.Variable(torch.LongTensor(tweet.cont_vec)))

        for i, w in enumerate(wembeds):
            cs = autograd.Variable(torch.LongTensor(tweet.cont_char_vec[i]))
            cembeds = self.char_embeddings(cs)
            clstm_out, self.chidden = self.clstm(
                cembeds.view(len(cs), 1, -1), self.chidden)
            w = torch.cat([w, clstm_out[-1].view(-1)])
            self.hidden = self.lstm(w, self.hidden)

        tag_scores = self.hidden2tag(self.hidden[0]).view(-1)
        return tag_scores

def train(tweets, words, chars, epochs):
    torch.manual_seed(1)
    model = LSTM(300, 300, 100, 100, len(words)+1, len(chars)+1, 11)
    loss_function = nn.MultiLabelSoftMarginLoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(epochs):
        losses = []
        for tweet in tweets:
            model.zero_grad()

            model.hidden = model.init_hidden()
            model.chidden = model.init_chidden()

            targets = autograd.Variable(torch.FloatTensor(tweet.emotions))

            tag_scores = model(tweet)

            loss = loss_function(tag_scores, targets)
            loss.backward()
            optimizer.step()
            losses.append(loss.data.mean())
        print('[%d/%d] Loss: %.3f' % (epoch+1, epochs, np.mean(losses)))
    return model



