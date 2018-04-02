import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Input
from keras.layers import concatenate
from keras.layers import Embedding
from keras.models import Model
from keras.preprocessing import sequence
import tensorflow as tf
numpy.random.seed(7)

class Twitter:

    def __init__(self, words_size, chars_size, max_tweet_length, max_word_length,
                w_embed_size, c_embed_size, clstm_hidden_size, 
                lstm_hidden_size, out_hidden_size, out_size):

        self.max_tweet_length = max_tweet_length

        words = Input(shape=(max_tweet_length,))
        words_vec = Embedding(words_size, w_embed_size, input_length=max_tweet_length)(words)


        c_seq = []

        chars = Input(shape=(max_tweet_length,max_word_length))
        for i in range(max_tweet_length):
            char = chars[i]
            char = Input(shape=(max_word_length,))
            char_vec = Embedding(chars_size, c_embed_size, input_length=max_word_length)(char)
            clstm = LSTM(clstm_hidden_size)(char_vec)
            c_seq.append(clstm)

        chars_vec = tf.reshape(tf.convert_to_tensor(c_seq), [-1, 50, 100])

        input_vec = concatenate([words_vec, chars_vec])

        print (input_vec)


        x = LSTM(lstm_hidden_size)(input_vec)
        x = Dense(out_hidden_size, activation='relu')(x)
        x = Dense(out_size, activation='sigmoid')(x)

        self.model = Model(inputs=[words, chars], outputs=x)

        self.model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
        

    def train(self, X_train, y_train, model):
        X_train = sequence.pad_sequences(X_train, maxlen=self.max_tweet_length)
        self.model.fit(X_train, y_train, epochs=30, batch_size=64)