import dynet_config
dynet_config.set(random_seed=1)

import dynet as dy
import numpy as np

import time
import pickle

class Twitter:

    def __init__(self, words_size, chars_size,
            w_embed_size, c_embed_size, clstm_hidden_size,
            lstm_hidden_size, lstm_num_layers,
            out_hidden_size, out_size):

        self.words_size = words_size
        self.chars_size = chars_size
        self.out_size = out_size

        self.w_embed_size = w_embed_size
        self.c_embed_size = c_embed_size
        self.clstm_hidden_size = clstm_hidden_size
        self.lstm_hidden_size = lstm_hidden_size * 2 # must be even
        self.lstm_num_layers = lstm_num_layers
        self.out_hidden_size = out_hidden_size

        self.model = dy.Model()
        self.trainer = dy.AdamTrainer(self.model)

        # words and tags, entities embeddings
        self.wlookup = self.model.add_lookup_parameters((words_size, self.w_embed_size))
        self.clookup = self.model.add_lookup_parameters((chars_size, self.c_embed_size))

        # feature extractor
        self.lstm = dy.LSTMBuilder(
                self.lstm_num_layers,
                self.w_embed_size +self.clstm_hidden_size,
                self.lstm_hidden_size,
                self.model,
        )

        # char encoder
        self.clstm = dy.LSTMBuilder(
                self.lstm_num_layers,
                self.c_embed_size,
                self.clstm_hidden_size,
                self.model,
        )
        self.char_to_lstm      = self.model.add_parameters((self.clstm_hidden_size, self.c_embed_size))
        self.char_to_lstm_bias = self.model.add_parameters((self.clstm_hidden_size))

        # transform word+pos vector into a vector similar to the lstm output
        # used to generate padding vectors
        self.word_to_lstm      = self.model.add_parameters((self.lstm_hidden_size, self.w_embed_size + self.clstm_hidden_size))
        self.word_to_lstm_bias = self.model.add_parameters((self.lstm_hidden_size))

        self.output_hidden      = self.model.add_parameters((self.out_hidden_size, self.lstm_hidden_size))
        self.output_hidden_bias = self.model.add_parameters((self.out_hidden_size))
        self.output      = self.model.add_parameters((self.out_size, self.out_hidden_size))
        self.output_bias = self.model.add_parameters((self.out_size))


    def save(self, name):
        params = (
            self.words_size, self.chars_size,
            self.w_embed_size,
            self.c_embed_size, self.clstm_hidden_size,
            self.lstm_hidden_size // 2, self.lstm_num_layers,
            self.out_hidden_size, self.out_size
        )
        # save model
        self.model.save(f'{name}.model')
        # save pickle
        with open(f'{name}.pickle', 'wb') as f:
            pickle.dump(params, f)

    @staticmethod
    def load(name):
        with open(f'{name}.pickle', 'rb') as f:
            params = pickle.load(f)
            parser = Twitter(*params)
            parser.model.populate(f'{name}.model')
            return parser

    def extract_features(self, tweet):
        inputs = []
        for i, entry in enumerate(tweet.cont_vec):
            c_seq = list()
            for c in tweet.cont_char_vec[i]:
                c_v = self.clookup[c]
                c_seq.append(c_v)
            c_vec = self.clstm.initial_state().transduce(c_seq)[-1]
            w_vec = self.wlookup[entry]
            i_vec = dy.concatenate([w_vec, c_vec])
            inputs.append(i_vec)
        outputs = self.lstm.initial_state().transduce(inputs)
        return outputs[-1]

    def forward(self, v):
        output_hidden = dy.tanh(self.output_hidden.expr() * v + self.output_hidden_bias.expr())
        output = dy.logistic(self.output.expr() * output_hidden + self.output_bias.expr())
        return output

    def train(self, tweets):
        losses = []
        loss_all = 0
        total_all = 0
        start_all = time.time()
        for i, tweet in enumerate(tweets):
            v = self.extract_features(tweet)
            output = self.forward(v)
            gold = tweet.emotions
            loss = dy.binary_log_loss(output, dy.inputTensor(gold))
            loss_all += loss.npvalue()[0]
            total_all += 1
            losses.append(loss)
        if len(losses) > 0:
            loss = dy.esum(losses)
            loss.scalar_value()
            loss.backward()
            self.trainer.update()
            dy.renew_cg()
        end = time.time()
        print(f'loss: {loss_all/total_all:.4f}\ttime: {end-start_all:,.2f} secs')

    def predict(self, tweet):
        v = self.extract_features(tweet)
        output = self.forward(v)
        prediction = np.where(output.npvalue() > 0.5, 1, 0)
        return prediction

            
