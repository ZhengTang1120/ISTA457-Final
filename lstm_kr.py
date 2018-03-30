import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Input
from keras.layers import concatenate
from keras.layers import Embedding
from keras.models import Model
numpy.random.seed(7)

def get_model(words_size, chars_size, max_tweet_length, max_word_length,
            w_embed_size, c_embed_size, clstm_hidden_size, 
            lstm_hidden_size, out_hidden_size, out_size):

    words = Input(shape=(words_size,))
    words_vec = Embedding(words_size, w_embed_size, input_length=max_tweet_length)(words)
    words_vec = Dense(max_tweet_length)(words_vec)
    print (words_vec)

    chars = Input(shape=(chars_size, max_tweet_length))
    chars_vec = Embedding(chars_size, c_embed_size, input_length=max_word_length)(chars)
    chars_vec = Dense(max_tweet_length)(chars_vec)
    clstm = LSTM(clstm_hidden_size)(chars_vec)
    print (clstm)

    input_vec = concatenate([words_vec, clstm])
    x = LSTM(lstm_hidden_size)(input_vec)
    x = Dense(out_hidden_size, activation='relu')(x)
    x = Dense(out_size, activation='sigmoid')(x)

    model = Model(inputs=[words, chars], outputs=[x])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    print(model.summary())
    return model
    # model.fit(X_train, y_train, nb_epoch=3, batch_size=64)
    # scores = model.evaluate(X_test, y_test, verbose=0)
    # print("Accuracy: %.2f%%" % (scores[1]*100))