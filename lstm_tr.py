import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time

class LSTM(nn.Module):

    def __init__(self, wembedding_dim, cembedding_dim, hidden_dim, chidden_dim, vocab_size, chars_size, hash_size, target_size):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.chidden_dim = chidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, wembedding_dim)
        self.char_embeddings = nn.Embedding(chars_size, cembedding_dim)
        self.hash_embeddings = nn.Embedding(hash_size, hidden_dim)

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

        hidden = self.hidden[0]
        if tweet.hashtags_vec != []:
            hembeds = self.hash_embeddings(autograd.Variable(torch.LongTensor(tweet.hashtags_vec)))
            h = torch.mean(hembeds, 0)
            hidden += h

        tag_scores = self.hidden2tag(hidden).view(-1)
        return tag_scores

def train(tweets, words, chars, hashtags, epochs):
    torch.manual_seed(1)
    model = LSTM(300, 50, 100, 10, len(words)+1, len(chars)+1, len(hashtags), 11)
    loss_function = nn.MultiLabelSoftMarginLoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(epochs):
        start = time.time()
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
        torch.save(model, "%d.model" % (epoch+1))
        end = time.time()
        print('[%d/%d] Loss: %.3f Time: %.2f' % (epoch+1, epochs, np.mean(losses), end-start))
    return model



